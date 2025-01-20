# coding: utf-8

import json
import logging
import os
import random
import re
from decimal import ROUND_HALF_EVEN, Decimal
from io import StringIO
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Sequence, Set, Tuple, Union
from uuid import uuid4

import cv2
import imgkit
import numpy as np
import pandas as pd
import pypandoc
import tqdm as TQDM
from matplotlib import pyplot as plt
from packaging.version import Version
from PIL import Image as PILImage

from latex_table_generator.base import (
    MatLike,
    crop_table_bbox,
    draw_table_bbox,
    fix_rotation_image,
    get_image,
    image_resize,
    paste_image_with_table_bbox,
    rotate_img_with_border,
    run_random_crop_rectangle,
)
from latex_table_generator.camelot_base import run_table_detect as run_table_detect_camelot
from latex_table_generator.errors import (
    ImagePasteError,
    NotColumnMatchError,
    NotLatexError,
    NotSupportMulticolumnLatexError,
    NotSupportMulticolumnLatexTableError,
)
from latex_table_generator.image2table import run_table_detect as run_table_detect_img2table

_support_merge_methods = ["horizontal", "vertical", "hybrid", "none"]
_latex_includegraphics_pattern = r"\\includegraphics{(.*.(?:jpg|png|JPG|PNG))}"
_latex_table_begin_pattern = r"\\begin{tabular}{.*}"
_latex_table_end_pattern = r"\\end{tabular}"
_default_css = r"""<style>
    table,
    th,
    td {{
        border: 1px solid black;
        border-collapse: collapse;
        font-size: 48px;
        font-weight: normal;
    }}

    table {{
        width: 100%;
    }}

    td {{
        padding-top: {padding}rem;
        padding-bottom: {padding}rem;
        padding-left: {padding}rem;
        padding-right: {padding}rem;
    }}
</style>"""
_random_headers = [
    [
        {"names": ["編號", "#"], "type": int, "empty": False, "hashtag": False, "sequence": True, "range": None, "choices": None},
        {
            "names": ["部位"],
            "type": str,
            "empty": False,
            "hashtag": False,
            "sequence": False,
            "range": None,
            "choices": [
                "樑",
                "柱",
                "版",
                "牆",
                "梯",
                "上版",
                "下版",
                "樑料",
                "柱料",
                "版料",
                "牆料",
                "梯料",
                "樑補",
                "柱補",
                "版補",
                "牆補",
                "梯補",
                "上版補",
                "下版補",
                "梯1",
                "梯2",
                "梯3",
                "電梯牆",
                "下層版筋",
            ],
        },
        {"names": ["號數"], "type": int, "empty": False, "hashtag": True, "sequence": False, "range": (1, 20), "choices": None},
        {
            "names": ["圖示", "加工形狀", "加工型狀", "加工形式", "加工型式", "形狀", "型狀", "形式", "型式"],
            "type": str,
            "empty": False,
            "hashtag": False,
            "sequence": False,
            "range": None,
            "choices": None,
        },
        {
            "names": ["長度", "長度(cm)", "總長度", "料長"],
            "type": int,
            "empty": False,
            "hashtag": False,
            "sequence": False,
            "range": (1, 2000),
            "choices": None,
        },
        {
            "names": ["數量", "支數"],
            "type": int,
            "empty": False,
            "hashtag": False,
            "sequence": False,
            "range": (1, 2000),
            "choices": None,
        },
        {
            "names": ["重量", "重量(kg)", "重量Kg", "重量噸"],
            "type": int,
            "empty": False,
            "hashtag": False,
            "sequence": False,
            "range": (1, 20000),
            "choices": None,
        },
        {"names": ["備註"], "type": str, "empty": True, "hashtag": False, "sequence": False, "range": None, "choices": None},
    ]
]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

# Check pypandoc version
if Version(pypandoc.get_pandoc_version()) < Version("3.1.2"):
    raise RuntimeError(
        "pandoc version need >= 3.1.2, please run `ARCH=x86_64 bash scripts/install-pandoc.sh && source ~/.bashrc`"
    )


def get_subfolder_path(
    path: PathLike,
    targets: Sequence[Union[str, Sequence[str]]] = (
        ("*.jpg", "*.png"),
        ("*.txt"),
    ),
) -> List[Path]:
    subfolder_paths = list()
    find_paths = [path]
    while find_paths:
        find_path = Path(find_paths.pop(0))
        logger.debug(f"get_subfolder_path: Search {find_path!s}")

        status = [False for _ in range(len(targets))]
        for i, target in enumerate(targets):
            if isinstance(target, str):
                if [p for p in Path(find_path).glob(target)]:
                    status[i] = True
            else:
                for _target in target:
                    if [p for p in Path(find_path).glob(_target)]:
                        status[i] = True

        if np.array(status).all():
            subfolder_paths.append(find_path)

        for folder in find_path.iterdir():
            if folder.is_dir():
                logger.debug(f"get_subfolder_path: Add {folder!s}")
                find_paths.append(folder)
    return subfolder_paths


def convert_latex_table_to_markdown(
    src: str,
) -> List[str]:
    html = pypandoc.convert_text(src, "html", format="latex")
    with StringIO(html) as f:
        dfs = pd.read_html(f)

    return [df.to_markdown(index=False).replace("nan", "   ") for df in dfs]


def preprocess_latex_table_string(
    latex_table_str: str,
) -> str:
    processed_latex_table_str = re.sub(_latex_table_begin_pattern, "", latex_table_str)
    processed_latex_table_str = re.sub(_latex_table_end_pattern, "", processed_latex_table_str)
    processed_latex_table_str = processed_latex_table_str.replace("\n", " ").strip()

    # Filter multi \hline error
    rows = processed_latex_table_str.split(r"\\")
    new_rows = list()
    for row in rows:
        _row = row
        if row.count(r"\hline") > 1:
            _row = _row.replace(r"\hline", "").strip()
            _row = rf"\hline {_row}"
        new_rows.append(_row)

    return "\\\\\n".join(new_rows)


def pre_check_latex_table_string(
    latex_table_str: str,
) -> Tuple[str, str]:
    results = re.findall(_latex_table_begin_pattern, latex_table_str)
    if not results:
        raise NotLatexError("Not latex table")
    elif "multicolumn" in latex_table_str:
        raise NotSupportMulticolumnLatexError("Not support convert have multicolumn latex")
    elif len(results) > 1:
        raise NotSupportMulticolumnLatexTableError("Not support convert have latex table")

    begin_str = results[0]
    end_str = r"\end{tabular}"
    return (begin_str, end_str)


def convert_latex_table_to_pandas(
    latex_table_str: str,
    headers: Union[bool, Sequence[str], None] = None,
) -> pd.DataFrame:
    processed_latex_table_str = preprocess_latex_table_string(latex_table_str)
    rows = [row.strip() for row in processed_latex_table_str.split(r"\\") if "&" in row]  # 過濾掉無關的行

    # 拆分表格各儲存格
    table_data = [row.replace(r"\\", "").replace(r"\hline", "").strip().split("&") for row in rows]
    cleaned_data = [[cell.strip() for cell in row] for row in table_data]

    if isinstance(headers, bool) and headers:
        headers = cleaned_data[0]  # 第一行是列名
        data = cleaned_data[1:]  # 剩餘的是數據
        df = pd.DataFrame(data, columns=headers)
    elif headers:
        df = pd.DataFrame(cleaned_data)
    return df


def random_generate_latex_table_string(
    headers: List[Dict[str, Any]],
    rows_range: Tuple[int, int],
    add_space_row_percentage: float,
    rng: random.Random = None,
) -> str:
    latex_table_str = rf"\begin{{tabular}}{{{'c'.join(['|' for _ in range(len(headers)+1)])}}}"

    # Add column name
    columns = list()
    for header in headers:
        names = header.get("names")
        columns.append(names[rng.randint(0, len(names) - 1)])
    latex_table_str += r"\hline " + " & ".join(columns) + " \\\\\n"

    # Add row
    rows_count = rng.randint(rows_range[0], rows_range[1])
    sequence_start_index = rng.randint(0, 100)
    rows = list()
    for i in range(rows_count):
        is_space_row = rng.randint(1, 100) <= int(add_space_row_percentage * 100) if add_space_row_percentage else False
        values = list()
        for header in headers:
            if is_space_row:
                values.append("")
            else:
                _type = header.get("type")
                _empty = header.get("empty", True)
                _hashtag = header.get("hashtag", False)
                _sequence = header.get("sequence", False)
                _range = header.get("range", None)
                _choices = header.get("choices", None)
                _type = getattr(__builtins__, _type) if isinstance(_type, str) else _type  # Get type class, ex: <class 'int'>

                value = None
                if isinstance(_empty, bool) and _empty and rng.randint(0, 1) == 1:
                    pass
                elif isinstance(_empty, float) and _empty > 0.0 and rng.randint(1, 100) <= int(_empty * 100):
                    pass
                elif issubclass(_type, (int, float)):
                    if _sequence:
                        value = sequence_start_index + i
                    elif _range:
                        value = rng.randint(_range[0], _range[1]) if issubclass(_type, int) else rng.uniform(_range[0], _range[1])
                    else:
                        value = rng.randint(0, 100) if issubclass(_type, int) else rng.uniform(0, 100)
                elif _choices and issubclass(_type, str):
                    value = _choices[rng.randint(0, len(_choices) - 1)]

                value = str(value) if value is not None else value
                value = rf"\#{value}" if value is not None and _hashtag else value
                values.append(value if value is not None else "")
        rows.append(r"\hline " + " & ".join(values))

    latex_table_str += "\\\\\n".join(rows)
    latex_table_str += "\\hline\n\\end{tabular}"
    return latex_table_str


def dropout_table_content(
    latex_table_str: str,
    dropout_percentage: float,
    rng: random.Random = None,
) -> str:
    """Dropout cell content

    Args:
        latex_table_str (str): latex table string
        dropout_table_content (float, optional): Dropout table content percentage.
        rng (random.Random, optional): random generator. Defaults to None.

    Returns:
        str: Dropped content latex table string
    """
    logger.debug("Run dropout_cell_content")

    if dropout_percentage <= 0.0:
        return latex_table_str

    (begin_str, end_str) = pre_check_latex_table_string(latex_table_str=latex_table_str)
    table = convert_latex_table_to_pandas(latex_table_str, headers=True)

    # Get dropout indexes
    dropout_indexes = list()
    for i in range(len(table)):
        for j, v in enumerate(table.iloc[i]):
            dropout_indexes.append((i, j))
    rng.shuffle(dropout_indexes)
    dropout_indexes = dropout_indexes[: int(len(dropout_indexes) * dropout_percentage)]

    # Start drop cell content
    rows = [r"\hline " + " & ".join([str(c) for c in table.columns])]
    for i in range(len(table)):
        contents = list()

        for j, v in enumerate(table.iloc[i]):
            if (i, j) in dropout_indexes:
                contents.append("")
            else:
                contents.append(v)
        contents_str = " & ".join(contents)
        rows.append(rf"\hline {contents_str}" if r"\cline" not in contents_str else contents_str)

    rows.append(r"\hline")
    final_latex_table_image_str = " \\\\\n".join(rows)
    return f"{begin_str}\n{final_latex_table_image_str}\n{end_str}"


def filling_image_to_cell(
    latex_table_str: str,
    image_paths: List[str],
    image_specific_headers: List[str],
    rng: random.Random = None,
) -> str:
    logger.debug("Run filling_image_to_cell")
    (begin_str, end_str) = pre_check_latex_table_string(latex_table_str=latex_table_str)

    table = convert_latex_table_to_pandas(latex_table_str, headers=True)
    rows = [r"\hline " + " & ".join([str(c) for c in table.columns])]
    col_names: List[Tuple[int, str, bool]] = list()
    for i, col_name in enumerate(table.columns):
        for specific_header in image_specific_headers:
            if re.match(specific_header, col_name.replace("状", "狀")):
                col_names.append((i, col_name))

    if not col_names:
        raise NotColumnMatchError(f"Can not find {image_specific_headers} column name")

    _image_paths = image_paths.copy()
    rng.shuffle(_image_paths)
    for i in range(len(table)):
        contents = list()
        col, col_name = col_names[0]
        logger.debug(f"col: {col}, name: {col_name}")

        is_space_row = sum([1 if v else 0 for v in table.iloc[i]]) == 0

        for j, v in enumerate(table.iloc[i]):  # 紀錄 cell 內容
            if j == col and not is_space_row:  # 若是是要替換 image 的欄位
                index = i % len(_image_paths)
                contents.append(rf"\includegraphics{{{image_paths[index]}}}")
            else:
                contents.append(v)
        contents_str = " & ".join(contents)
        rows.append(rf"\hline {contents_str}" if r"\cline" not in contents_str else contents_str)

    rows.append(r"\hline")
    final_latex_table_image_str = " \\\\\n".join(rows)
    return f"{begin_str}\n{final_latex_table_image_str}\n{end_str}"


def merge_horizontal_cell(
    latex_table_str: str,
    rng: random.Random = None,
    count: int = 1,
    contents: Union[Sequence[str], str] = None,
    horizontal: Union[int, Tuple[int, int]] = 1,
    **kwds,
) -> Tuple[str, str]:
    # TODO: 支援可以少數欄水平合併
    logger.debug("Run merge_horizontal_cell")
    (begin_str, end_str) = pre_check_latex_table_string(latex_table_str=latex_table_str)
    processed_latex_table_str = preprocess_latex_table_string(latex_table_str)

    rows_image = [s.strip() for s in processed_latex_table_str.split(r"\\") if s.strip() and "&" in s]
    rows_label = [s.strip() for s in processed_latex_table_str.split(r"\\") if s.strip() and "&" in s]
    rand_nums = [i for i in range(1, len(rows_image))]
    rng.shuffle(rand_nums)
    rand_nums = rand_nums[:count]

    for rand_num in rand_nums:  # 執行 count 次
        texts = rows_image[rand_num].replace(r"\hline", "").strip().split("&")
        texts_str = contents[rng.randint(0, len(contents) - 1)] if contents else texts[rng.randint(0, len(texts) - 1)]
        texts_str_label = "&".join([texts_str for _ in range(len(texts))])
        rows_image[rand_num] = rf"\hline \multicolumn{{{len(texts)}}}{{|c|}}{{{texts_str}}}"
        rows_label[rand_num] = rf"\hline {texts_str_label}"

    rows_image.append(r"\hline")
    rows_label.append(r"\hline")
    latex_table_image_str = "\\\\\n".join(rows_image)
    latex_table_label_str = "\\\\\n".join(rows_label)
    return (
        f"{begin_str}\n{latex_table_image_str}\n{end_str}",
        f"{begin_str}\n{latex_table_label_str}\n{end_str}",
    )


def merge_vertical_cell(
    latex_table_str: str,
    rng: random.Random = None,
    count: int = 1,
    contents: Union[Sequence[str], str] = None,
    vertical: Union[int, Tuple[int, int]] = 1,
    specific_headers: Sequence[str] = None,
    **kwds,
) -> Tuple[str, str]:
    logger.debug("Run merge_vertical_cell")
    (begin_str, end_str) = pre_check_latex_table_string(latex_table_str=latex_table_str)
    end_str = r"\end{tabular}"

    table = convert_latex_table_to_pandas(latex_table_str, headers=True)
    rows_image = [r"\hline " + " & ".join([str(c) for c in table.columns])]
    rows_label = [r"\hline " + " & ".join([str(c) for c in table.columns])]
    rand_nums = [i for i in range(0, len(table) - 1, 2)]
    rng.shuffle(rand_nums)
    rand_nums = sorted(rand_nums[:count])
    logger.debug(f"rand_nums: {rand_nums}")
    col_names = list()
    if specific_headers:
        for i, col_name in enumerate(table.columns):
            for specific_header in specific_headers:
                if re.match(specific_header, col_name):
                    col_names.append((i, col_name))
    else:
        col_names = [(i, col_name) for i, col_name in enumerate(table.columns)]

    if not col_names:
        raise NotColumnMatchError(f"Can not find {specific_headers} column name")

    # 預先決定要合併的列
    multirow_index = list()
    all_multirow_index = list()
    all_col_names = list()
    all_cell_contents = list()
    for rand_num_index, rand_num in enumerate(rand_nums):
        while True:
            # 檢查 vertical
            if isinstance(vertical, int):
                multirow_num = vertical + 1
            elif isinstance(vertical, (tuple, list)) and len(vertical) >= 2:
                multirow_num = rng.randint(vertical[0], min(vertical[1], len(table) - rand_num)) + 1
            else:
                raise TypeError(f"vertical should be int or tuple. But got '{vertical}'")

            if multirow_num + rand_num <= len(table) and (
                rand_num_index == len(rand_nums) - 1 or multirow_num + rand_num <= rand_nums[rand_num_index + 1]
            ):
                index = [rand_num + n for n in range(multirow_num)]
                multirow_index.append(index)
                all_multirow_index.extend(index)
                rng.shuffle(col_names)
                all_col_names.append(col_names[0])
                all_cell_contents.append(contents[rng.randint(0, len(contents) - 1)] if contents else None)
                break

    for i in range(len(table)):
        contents_image = list()
        contents_label = list()
        if i in all_multirow_index:  # add multirow and cline
            multirow_data = np.array([multirow.index(i) if i in multirow else -1 for multirow in multirow_index])
            index = np.nonzero(multirow_data + 1)[0][0]
            col, col_name = all_col_names[index]
            logger.debug(f"col: {col}, name: {col_name}")

            for j, v in enumerate(table.iloc[i]):  # 紀錄 cell 內容
                if j == col:  # 若是是要 multirow 的欄位
                    multirow_num = len(multirow_index[index])
                    rand_num = rand_nums[index]
                    logger.debug(f"multirow_num: {multirow_num}")

                    _cell_contents = table[table.columns[col]].iloc[rand_num : rand_num + multirow_num].values
                    _cell_content = (
                        all_cell_contents[index]
                        if all_cell_contents[index]
                        else _cell_contents[rng.randint(0, len(_cell_contents) - 1)]
                    )
                    contents_label.append(f"**{multirow_num} {_cell_content}")
                    if i in rand_nums:  # multirow 不會重複加, 所以只加第一次
                        contents_image.append(rf"\multirow{{{multirow_num}}}{{*}}{{{_cell_content}}}")
                    else:
                        contents_image.append("")
                else:
                    contents_image.append(v)
                    contents_label.append(v)
        else:
            contents_image = [str(e) for e in table.iloc[i]]
            contents_label = [str(e) for e in table.iloc[i]]
        contents_image = " & ".join(contents_image)
        contents_label = " & ".join(contents_label)
        rows_image.append(rf"\hline {contents_image}" if r"\cline" not in contents_image else contents_image)
        rows_label.append(rf"\hline {contents_label}" if r"\cline" not in contents_label else contents_label)

    rows_image.append(r"\hline")
    rows_label.append(r"\hline")
    final_latex_table_image_str = " \\\\\n".join(rows_image)
    final_latex_table_label_str = " \\\\\n".join(rows_label)
    return (
        f"{begin_str}\n{final_latex_table_image_str}\n{end_str}",
        f"{begin_str}\n{final_latex_table_label_str}\n{end_str}",
    )


def merge_vertical_and_horizontal_cell(
    latex_table_str: str,
    rng: random.Random = None,
    count: int = 1,
    contents: Union[Sequence[str], str] = None,
    vertical: Union[int, Tuple[int, int]] = 1,
    horizontal: Union[int, Tuple[int, int]] = 1,
    **kwds,
) -> Tuple[str, str]:
    logger.debug("Run merge_vertical_and_horizontal_cell")
    (begin_str, end_str) = pre_check_latex_table_string(latex_table_str=latex_table_str)
    end_str = r"\end{tabular}"

    table = convert_latex_table_to_pandas(latex_table_str, headers=True)
    rows_image = [r"\hline " + " & ".join([str(c) for c in table.columns])]
    rows_label = [r"\hline " + " & ".join([str(c) for c in table.columns])]
    rand_nums = [i for i in range(0, len(table) - 1, 2)]
    rng.shuffle(rand_nums)
    rand_nums = sorted(rand_nums[:count])
    logger.debug(f"rand_nums: {rand_nums}")

    col_names = list()
    for i, col_name in enumerate(table.columns):
        # 檢查 horizontal
        if isinstance(horizontal, int):
            col_span = horizontal + 1
        elif isinstance(horizontal, (tuple, list)) and len(horizontal) >= 2:
            col_span = rng.randint(horizontal[0], min(horizontal[1], len(table.columns))) + 1
        else:
            raise TypeError(f"horizontal should be int or tuple. But got '{horizontal}'")
        col_names.append(([i + col for col in range(col_span)], col_name))

    # 預先決定 multirow 要合併的列
    multirow_index = list()
    all_multirow_index = list()
    all_col_names = list()
    all_cell_contents = list()
    for rand_num_index, rand_num in enumerate(rand_nums):
        while True:
            # 檢查 vertical
            if isinstance(vertical, int):
                row_span = vertical + 1
            elif isinstance(vertical, (tuple, list)) and len(vertical) >= 2:
                row_span = rng.randint(vertical[0], min(vertical[1], len(table) - rand_num)) + 1
            else:
                raise TypeError(f"vertical should be int or tuple. But got '{vertical}'")

            if row_span + rand_num <= len(table) and (
                rand_num_index == len(rand_nums) - 1 or row_span + rand_num <= rand_nums[rand_num_index + 1]
            ):
                index = [rand_num + n for n in range(row_span)]
                multirow_index.append(index)
                all_multirow_index.extend(index)
                rng.shuffle(col_names)
                all_col_names.append(col_names[0])
                all_cell_contents.append(contents[rng.randint(0, len(contents) - 1)])
                break

    for i in range(len(table)):
        contents_image = list()
        contents_label = list()
        if i in all_multirow_index:  # add multirow and cline
            multirow_data = np.array([multirow.index(i) if i in multirow else -1 for multirow in multirow_index])
            index = np.nonzero(multirow_data + 1)[0][0]
            col, col_name = all_col_names[index]
            logger.debug(f"col: {col}, name: {col_name}")

            for j, v in enumerate(table.iloc[i]):  # 紀錄 cell 內容
                if j in col:  # 若是是要 multirow 的欄位
                    row_span = len(multirow_index[index])
                    rand_num = rand_nums[index]
                    logger.debug(f"row_span: {row_span}")

                    _cell_content = all_cell_contents[index]
                    contents_label.append(f"**{row_span} {_cell_content}")
                    if i in rand_nums:  # multirow 不會重複加, 所以只加第一次
                        if j == col[0]:
                            contents_image.append(
                                rf"\multicolumn{{{len(col)}}}{{|c|}}{{\multirow{{{row_span}}}{{*}}{{{_cell_content}}}}}"
                            )
                        else:
                            pass

                    else:
                        contents_image.append("")
                else:
                    contents_image.append(v)
                    contents_label.append(v)
        else:
            contents_image = [str(e) for e in table.iloc[i]]
            contents_label = [str(e) for e in table.iloc[i]]
        contents_image = " & ".join(contents_image)
        contents_label = " & ".join(contents_label)
        rows_image.append(rf"\hline {contents_image}" if r"\cline" not in contents_image else contents_image)
        rows_label.append(rf"\hline {contents_label}" if r"\cline" not in contents_label else contents_label)

    rows_image.append(r"\hline")
    rows_label.append(r"\hline")
    final_latex_table_image_str = " \\\\\n".join(rows_image)
    final_latex_table_label_str = " \\\\\n".join(rows_label)
    return (
        f"{begin_str}\n{final_latex_table_image_str}\n{end_str}",
        f"{begin_str}\n{final_latex_table_label_str}\n{end_str}",
    )


def latex_table_to_image(
    latex_table_str: str,
    css: str,
    format: str = "jpeg",
    quality: Union[str, int] = "100",
    width: Union[str, int] = 2048,
    **kwds,
) -> PILImage.Image:
    html = css.format(**kwds) + "\n" + pypandoc.convert_text(latex_table_str, "html", format="latex")
    logger.debug(html)
    with TemporaryDirectory(prefix="latex_temp") as temp_dir:
        try:
            imgkit.from_string(
                html,
                str(Path(temp_dir, "temp.jpg")),
                options={
                    "format": format,
                    "quality": str(quality),
                    "width": str(width),
                    "quiet": "",
                    "enable-local-file-access": "",
                },
            )
            image = PILImage.open(Path(temp_dir, "temp.jpg"))
            return image
        except Exception as e:
            logger.exception(e)
            logger.error("ERROR: latex_table_to_image")
    return None


def paste_fit_size_latex_table_to_image(
    latex_table_strs: List[str],
    file_image: PILImage.Image,
    position: Tuple[int, int, int, int],
    css: str,
    skew_angle: float,
    format: str = "jpeg",
    quality: Union[str, int] = "100",
    max_paddings: float = 3.0,
    step: float = -0.2,
    paste_vertical: bool = False,
) -> Tuple[PILImage.Image, List[Tuple[int, int, int, int]]]:
    """Paste table image to passed image, will auto calc generate image size.

    Args:
        latex_table_strs (List[str]): latex table string
        file_image (PILImage.Image): base image
        position (Tuple[int, int, int, int]): can paste image area position, (x1, y1, x2, y2)
        css (str): generate image's css
        skew_angle (float): generate image skew angle
        format (str, optional): generate image format. Defaults to "jpeg".
        quality (Union[str, int], optional): generate image quality. Defaults to "100".
        max_paddings (float, optional): generate image table cell max padding. Defaults to 3.0.
        step (float, optional): generate image table cell padding iterate step. like default is [3.0, 2.8, ..., 0.2, 0.0]. Defaults to -0.2.
        paste_vertical (bool, optional): paste image is vertical. Defaults to False.

    Returns:
        Tuple[PILImage.Image, List[Tuple[int, int, int, int]]]: (pasted image, table position: [(x1, y1, x2, y2)])
    """
    file_image: MatLike = get_image(src=file_image)
    step = step if step < 0 else (-1) * step
    area_width = int(position[2] - position[0])
    area_height = int(file_image.shape[0] - position[1])

    final_image = None
    table_positions = list()
    for index, latex_table_str in enumerate(latex_table_strs):
        (image_width, image_height) = (area_width, area_height)
        if paste_vertical:
            image_height = image_height // len(latex_table_strs)
        else:
            image_width = image_width // len(latex_table_strs)
        for padding in sorted(
            [float(Decimal(p).quantize(Decimal(".1"), rounding=ROUND_HALF_EVEN)) for p in np.arange(max_paddings, 0.0, step)]
            + [0.0],
            reverse=True,
        ):
            image = get_image(
                src=latex_table_to_image(
                    latex_table_str,
                    width=image_width,
                    css=css,
                    format=format,
                    quality=str(quality),
                    padding=padding,
                )
            )
            image = rotate_img_with_border(img=image, angle=skew_angle)

            if image.shape[1] > image_width:
                image = image_resize(src=image)

            if image.shape[0] > image_height:
                continue

            try:
                if paste_vertical:
                    (area_offset_x, area_offset_y) = (0, area_height * (index / len(latex_table_strs)))
                else:
                    (area_offset_x, area_offset_y) = (area_width * (index / len(latex_table_strs)), 0)
                logger.debug(f"position: {(int(position[0] + area_offset_x), int(position[1] + area_offset_y))}")
                logger.debug(f"image.shape: {image.shape}")
                logger.debug(f"area.shape: ({area_height}, {area_width})")
                (final_image, table_position) = paste_image_with_table_bbox(
                    src=final_image if final_image is not None else file_image,
                    dst=image,
                    position=(int(position[0] + area_offset_x), int(position[1] + area_offset_y)),
                )
                table_positions.append(table_position)
                break
            except ValueError as e:
                raise ImagePasteError("Can't paste image") from e
    return final_image, table_positions


def merge_cell(
    latex_table_str: str,
    merge_methods: Union[Set[str], Sequence[str]],
    h_contents: List[str],
    v_contents: List[str],
    vh_contents: List[str],
    specific_headers: List[str],
    vertical: Union[int, Tuple[int, int]],
    horizontal: Union[int, Tuple[int, int]],
    vertical_count: Union[int, Tuple[int, int]],
    horizontal_count: Union[int, Tuple[int, int]],
    rng: random.Random = None,
):
    __error = ValueError(f"merge_methods have not support method, should be choice from {_support_merge_methods}")
    for _merge_method in merge_methods:
        if _merge_method not in _support_merge_methods:
            raise __error

    if len(merge_methods) > 1:
        merge_method = merge_methods[rng.randint(0, len(merge_methods) - 1)]
    elif len(merge_methods) == 1:
        merge_method = merge_methods[0]
    else:
        raise __error

    if merge_method == "vertical":
        (latex_table_image_str, latex_table_label_str) = merge_vertical_cell(
            latex_table_str=latex_table_str,
            rng=rng,
            contents=v_contents,
            specific_headers=specific_headers,
            vertical=vertical,
            count=vertical_count if isinstance(vertical_count, int) else rng.randint(vertical_count[0], vertical_count[1]),
        )
    elif merge_method == "horizontal":
        (latex_table_image_str, latex_table_label_str) = merge_horizontal_cell(
            latex_table_str=latex_table_str,
            rng=rng,
            contents=h_contents,
            count=horizontal_count
            if isinstance(horizontal_count, int)
            else rng.randint(horizontal_count[0], horizontal_count[1]),
        )
    elif merge_method == "hybrid":
        (latex_table_image_str, latex_table_label_str) = merge_vertical_and_horizontal_cell(
            latex_table_str=latex_table_str,
            rng=rng,
            contents=vh_contents,
            vertical=vertical,
            horizontal=horizontal,
        )
    else:
        (latex_table_image_str, latex_table_label_str) = (latex_table_str, latex_table_str)
    return (latex_table_image_str, latex_table_label_str)


def main(
    output_path: PathLike,
    input_path: PathLike = None,
    merge_methods: Union[Sequence[str], Set[str]] = {"horizontal", "vertical", "hybrid", "none"},
    h_contents: List[str] = ["開口補強"],
    v_contents: List[str] = ["彎鉤", "鋼材筋"],
    vh_contents: List[str] = ["開口補強", "鋼材筋"],
    specific_headers: List[str] = [".*備註.*"],
    vertical: Union[int, Tuple[int, int]] = [1, 5],
    horizontal: Union[int, Tuple[int, int]] = [1, 5],
    vertical_count: Union[int, Tuple[int, int]] = [1, 3],
    horizontal_count: Union[int, Tuple[int, int]] = [1, 3],
    skew_angle: Union[int, Tuple[int, int]] = [-5, 5],
    image_paths: List[str] = None,
    image_specific_headers: List[str] = [".*圖示.*", ".*(?:加工)?[形型][狀式].*"],
    css: str = _default_css,
    count: int = 100,
    new_image_size: Tuple[int, int] = (2480, 3508),
    min_crop_size: Union[float, int] = None,
    rows_range: Tuple[int, int] = (1, 20),
    format: Set[str] = {"all"},
    multi_table: int = None,
    multi_table_paste_vertical: str = "none",
    html_label_cell_merge: bool = False,
    add_space_row_percentage: float = 0.3,
    dropout_percentage: float = None,
    tqdm: bool = True,
    **kwds,
):
    assert input_path is not None or (count is not None and count > 0), "Need pass 'input_path' or 'count'"

    format = set(format) if isinstance(format, (list, tuple)) else format
    rng = random.Random(kwds.get("seed", os.environ.get("SEED", None)))
    logger.setLevel(kwds.get("log_level", os.environ.get("LOG_LEVEL", "INFO")))
    full_random_generate = False

    # Create iter_data
    if input_path and Path(input_path).exists():
        iter_data = [d for d in Path(input_path).glob(r"*.txt")]
        iter_data = TQDM.tqdm(iter_data, desc=str(input_path)) if tqdm else iter_data
    else:
        full_random_generate = True
        iter_data = list()
        while len(iter_data) < count:
            file_id = str(uuid4())
            if file_id not in iter_data:
                iter_data.append(Path(file_id))
        iter_data = TQDM.tqdm(iter_data, desc="Full random generate") if tqdm else iter_data
    logger.debug(input_path) if tqdm else logger.info(input_path)

    # Create output path
    Path(output_path).mkdir(exist_ok=True, parents=True)
    if format & {"all"} or len(format) > 1:
        output_path_images = Path(output_path, "images")
        output_path_markdown = Path(output_path, "markdown")
        output_path_latex = Path(output_path, "latex")
        output_path_html = Path(output_path, "html")
        output_path_table_info = Path(output_path, "table_info")
        output_path_images.mkdir(exist_ok=True, parents=True)  # Always create image folder

        if "all" in format or "markdown" in format:
            output_path_markdown.mkdir(exist_ok=True, parents=True)
        if "all" in format or "latex" in format:
            output_path_latex.mkdir(exist_ok=True, parents=True)
        if "all" in format or "html" in format:
            output_path_html.mkdir(exist_ok=True, parents=True)
        if "all" in format or "table_info" in format:
            output_path_table_info.mkdir(exist_ok=True, parents=True)

    for index, filename in enumerate(iter_data):
        if multi_table_paste_vertical in ["random"]:
            paste_vertical = rng.randint(0, 1) == 1
        elif multi_table_paste_vertical in ["always", "none"]:
            paste_vertical = multi_table_paste_vertical == "always"
        else:
            raise ValueError("multi_table_paste_vertical value error, should be choice from ['random', 'always', 'none']")

        # Get file_image and latex_table_str
        if not full_random_generate:
            if Path(input_path, f"{filename.stem}.jpg").exists():
                file_image = PILImage.open(Path(input_path, f"{filename.stem}.jpg"))
            elif Path(input_path, f"{filename.stem}.png").exists():
                file_image = PILImage.open(Path(input_path, f"{filename.stem}.png"))
            else:
                logger.info(f"Not found {filename}.jpg or {filename}.png, skip file")
                continue
            file_image = get_image(src=file_image)
            (file_image, _) = fix_rotation_image(img=file_image)

            logger.info(f"Run [{index+1}/{len(iter_data)}] {filename.name}") if not tqdm else None

            with filename.open("r", encoding="utf-8") as f:
                latex_table_str = f.read()
                latex_table_strs = [latex_table_str for _ in range(multi_table if multi_table else 1)]
        else:
            file_image = PILImage.new(mode="RGB", size=new_image_size, color=(255, 255, 255))
            file_image = get_image(src=file_image)
            latex_table_strs = [
                random_generate_latex_table_string(
                    headers=_random_headers[0],
                    rows_range=rows_range,
                    add_space_row_percentage=add_space_row_percentage,
                    rng=rng,
                )
                for _ in range(multi_table if multi_table else 1)
            ]
            logger.info(f"Run [{index+1}/{len(iter_data)}]") if not tqdm else None
        logger.debug(f"latex_table_strs: {latex_table_strs}")

        try:
            # Fill image
            latex_table_strs = [
                filling_image_to_cell(
                    latex_table_str=latex_table_str,
                    rng=rng,
                    image_paths=image_paths,
                    image_specific_headers=image_specific_headers,
                )
                for latex_table_str in latex_table_strs
            ]

            # Dropout table cell content
            latex_table_strs = [
                dropout_table_content(
                    latex_table_str=latex_table_str,
                    rng=rng,
                    dropout_percentage=dropout_percentage,
                )
                for latex_table_str in latex_table_strs
            ]

            latex_table_merged_strs = [
                merge_cell(
                    latex_table_str=latex_table_str,
                    merge_methods=merge_methods,
                    h_contents=h_contents,
                    v_contents=v_contents,
                    vh_contents=vh_contents,
                    specific_headers=specific_headers,
                    vertical=vertical,
                    horizontal=horizontal,
                    vertical_count=vertical_count,
                    horizontal_count=horizontal_count,
                    rng=rng,
                )
                for latex_table_str in latex_table_strs
            ]

            logger.debug(latex_table_merged_strs)

            # Get table position
            if not full_random_generate:
                tables = run_table_detect_camelot(file_image)
                tables = run_table_detect_img2table(file_image) if not tables else tables
            else:
                tables = run_random_crop_rectangle(file_image, min_crop_size=min_crop_size, rng=rng)

            if tables:
                [  # Check merged latex table is correct
                    convert_latex_table_to_pandas(
                        latex_table_str=latex_table_merged_str[1],
                        headers=True,
                    )
                    for latex_table_merged_str in latex_table_merged_strs
                ]
                hollow_image = draw_table_bbox(src=file_image, tables=tables, margin=5)

                _skew_angle = skew_angle if isinstance(skew_angle, (int, float)) else rng.uniform(skew_angle[0], skew_angle[1])
                (final_image, table_positions) = paste_fit_size_latex_table_to_image(
                    latex_table_strs=[latex_table_merged_str[0] for latex_table_merged_str in latex_table_merged_strs],
                    file_image=hollow_image,
                    position=(tables[0].bbox.x1, tables[0].bbox.y1, tables[0].bbox.x2, tables[0].bbox.y2),
                    css=css,
                    skew_angle=_skew_angle,
                    paste_vertical=paste_vertical,
                )
                if final_image is None:
                    raise ImagePasteError("Can't paste image")

                # Convert image to label
                latex_table_image_results = [latex_table_merged_str[0] for latex_table_merged_str in latex_table_merged_strs]
                latex_table_label_results = [latex_table_merged_str[1] for latex_table_merged_str in latex_table_merged_strs]
                for i in range(len(latex_table_image_results)):
                    for r in re.finditer(_latex_includegraphics_pattern, latex_table_image_results[i]):
                        path = Path(str(r.group(1)))
                        with Path(path.parent.resolve(), path.stem + ".txt").open("r", encoding="utf-8") as f:
                            label = f.read()
                        latex_table_image_results[i] = re.sub(
                            str(r.group(0)).replace("\\", "\\\\"), label, latex_table_image_results[i]
                        )
                for i in range(len(latex_table_label_results)):
                    for r in re.finditer(_latex_includegraphics_pattern, latex_table_label_results[i]):
                        path = Path(str(r.group(1)))
                        with Path(path.parent.resolve(), path.stem + ".txt").open("r", encoding="utf-8") as f:
                            label = f.read()
                        latex_table_label_results[i] = re.sub(
                            str(r.group(0)).replace("\\", "\\\\"), label, latex_table_label_results[i]
                        )

                # Save label
                table_info = json.dumps(
                    {
                        "positions": table_positions,
                        "skew_angle": _skew_angle,
                    }
                )
                if format & {"markdown", "latex", "table_info"} and len(format) == 1:
                    plt.imsave(Path(output_path, filename.stem + ".jpg"), final_image)
                    with Path(output_path, filename.stem + ".txt").open("w", encoding="utf-8") as f:
                        if "markdown" in format:
                            markdown_tables = list()
                            for latex_table_result in latex_table_label_results:
                                markdown_tables.append(convert_latex_table_to_markdown(src=latex_table_result)[0])
                            f.write("\n\n".join(markdown_tables))

                        elif "latex" in format:
                            f.write("\n".join(latex_table_label_results))

                        elif "html" in format:
                            html_tables = list()
                            for latex_table_result in (
                                latex_table_image_results if html_label_cell_merge else latex_table_label_results
                            ):
                                html_tables.append(pypandoc.convert_text(latex_table_result, "html", format="latex"))
                            f.write("\n".join(html_tables))

                        elif "table_info" in format:
                            f.write(table_info)
                        else:
                            raise ValueError(f"format value error, got unknown format: {format}")
                elif format & {"all"} or len(format) > 1:
                    # Save image
                    plt.imsave(Path(output_path_images, filename.stem + ".jpg"), final_image)

                    # Save latex
                    if "all" in format or "latex" in format:
                        with Path(output_path_latex, filename.stem + ".txt").open("w", encoding="utf-8") as f:
                            f.write("\n".join(latex_table_label_results))

                    # Save markdown
                    if "all" in format or "markdown" in format:
                        with Path(output_path_markdown, filename.stem + ".txt").open("w", encoding="utf-8") as f:
                            markdown_tables = list()
                            for latex_table_result in latex_table_label_results:
                                markdown_tables.append(convert_latex_table_to_markdown(src=latex_table_result)[0])
                            f.write("\n\n".join(markdown_tables))

                    # Save html
                    if "all" in format or "html" in format:
                        with Path(output_path_html, filename.stem + ".txt").open("w", encoding="utf-8") as f:
                            html_tables = list()
                            for latex_table_result in (
                                latex_table_image_results if html_label_cell_merge else latex_table_label_results
                            ):
                                html_tables.append(pypandoc.convert_text(latex_table_result, "html", format="latex"))
                            f.write("\n".join(html_tables))

                    # Save table position
                    if "all" in format or "table_info" in format:
                        with Path(output_path_table_info, filename.stem + ".txt").open("w", encoding="utf-8") as f:
                            f.write(table_info)
                else:
                    raise ValueError(f"format value error, got unknown format: {format}")

            else:
                logger.info(f"Not detect table, so skip {filename.name}")
        except Exception as e:
            logger.exception(e)
            logger.error(f"{filename.name} have error")


if __name__ == "__main__":
    rng = random.Random(os.environ.get("SEED", None))
    latex_table_str = r"""\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|}
    \hline 編號 & 組編號 & 號數 & 長A & 型狀/長度B & 長C & 總長度 & 支數 & 重量 & 備註 \\
    \hline 1 & 彎料 & \#5 & & 450 & & 480 & 10 & 75 & \\
    \hline 2 & 彎料 & \#4 & 80 & 80 & & 340 & 720 & 2433 & \\
    \hline 3 & 彎料 & \#4 & 65 & 80 & & 310 & 10 & 31 & \\
    \hline 4 & 彎料 & \#4 & 10 & 81 & 12 & 105 & 2800 & 2922 & \\
    \hline 5 & 彎料 & \#4 & 10 & 81 & 12 & 105 & 2800 & 2922 & \\
    \hline 6 & 彎料 & \#4 & 10 & 81 & 12 & 105 & 2800 & 2922 & \\
    \hline 7 & 彎料 & \#4 & 10 & 81 & 12 & 105 & 2800 & 2922 & \\
    \hline 8 & 彎料 & \#4 & 10 & 81 & 12 & 105 & 2800 & 2922 & \\
    \hline 9 & 彎料 & \#4 & 10 & 81 & 12 & 105 & 2800 & 2922 & \\
    \hline 10 & 彎料 & \#4 & 10 & 81 & 12 & 105 & 2800 & \includegraphics{/mnt/d/Dprogram/latex-table-generator/steel.png} & \\
    \hline
    \end{tabular}"""

    latex_table_str_ = r"""\begin{tabular}{|c|c|c|}
\hline
\multirow{3}{*}{合併的儲存格} & \multicolumn{2}{|c|}{跨兩列的儲存格} \\
\cline{2-3} & 第一列 & 第二列 \\
\cline{2-3} & 第一列a & 第二列b \\
\hline
\end{tabular}"""

    latex_table_image_str, latex_table_label_str = merge_vertical_and_horizontal_cell(
        latex_table_str, rng=rng, contents=["以下空白"], count=3, vertical=(1, 4)
    )
    logger.debug(latex_table_image_str)
    logger.debug(latex_table_label_str)

    Path("./outputs").mkdir(exist_ok=True)

    image = latex_table_to_image(
        latex_table_str=latex_table_image_str,
        css=_default_css,
        padding=2,
    )
    if image:
        image.save("outputs/origin-output.png")
        tables = run_table_detect_camelot(image)
        crop_table_images = crop_table_bbox(src=image, tables=tables, margin=10)
        if crop_table_images:
            plt.imsave("outputs/output.png", crop_table_images[0])

    image = latex_table_to_image(
        latex_table_str=latex_table_label_str,
        css=_default_css,
        padding=2,
    )
    if image:
        image.save("outputs/origin-label.png")
        tables = run_table_detect_camelot(image)
        crop_table_images = crop_table_bbox(src=image, tables=tables, margin=10)
        if crop_table_images:
            plt.imsave("outputs/label.png", crop_table_images[0])
