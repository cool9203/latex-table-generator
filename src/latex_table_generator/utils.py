# coding: utf-8

import ast
import json
import logging
import re
from os import PathLike
from pathlib import Path
from typing import Any, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from latex_table_generator.errors import (
    NotLatexError,
    NotSupportLatexError,
    NotSupportMulticolumnLatexError,
    NotSupportMultiLatexTableError,
)

_latex_table_begin_pattern = r"\\begin{tabular}{.*}"
_latex_table_end_pattern = r"\\end{tabular}"


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


def get_subfolder_path(
    path: PathLike,
    targets: Sequence[Union[str, Sequence[str]]] = (
        ("*.jpg", "*.png"),
        ("*.txt",),
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
        raise NotSupportMultiLatexTableError("Not support convert have multi latex table")

    begin_str = results[0]
    end_str = r"\end{tabular}"
    return (begin_str, end_str)


def convert_latex_table_to_pandas(
    latex_table_str: str,
    headers: Union[bool, Sequence[str], None] = None,
) -> pd.DataFrame:
    pre_check_latex_table_string(latex_table_str=latex_table_str)
    processed_latex_table_str = preprocess_latex_table_string(latex_table_str)
    rows = [row.strip() for row in processed_latex_table_str.split(r"\\") if "&" in row]  # 過濾掉無關的行

    # 拆分表格各儲存格
    table_data = [row.replace(r"\\", "").replace(r"\hline", "").strip().split("&") for row in rows]
    cleaned_data = [[cell.strip() for cell in row] for row in table_data]

    try:
        if isinstance(headers, bool) and headers:
            headers = cleaned_data[0]  # 第一行是列名
            data = cleaned_data[1:]  # 剩餘的是數據
            df = pd.DataFrame(data, columns=headers)
        elif headers:
            df = pd.DataFrame(cleaned_data)
    except ValueError as e:
        raise NotSupportLatexError("Not support this latex") from e
    return df


def load_render_header_file(
    path: PathLike,
) -> Any:
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as f:
            content = f.read()
            if path.suffix in [".py"]:
                content = ast.literal_eval(content)
            elif path.suffix in [".json"]:
                content = json.loads(content)
    except ValueError as e:
        if "malformed node or string" in str(e):
            raise ValueError("Type need be string of type, not direct pass type class") from e
        raise e
    return content
