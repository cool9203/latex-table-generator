# coding: utf-8

import logging
import os
import random
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence, Tuple, Union

import pandas as pd
from matplotlib import pyplot as plt
from pdf2image import convert_from_path
from PIL import Image as PILImage

from latex_table_generator.base import crop_table_bbox
from latex_table_generator.camelot_base import run_table_detect

_latex_table_begin_pattern = r"\\begin{tabular}{.*}"
_latex_table_end_pattern = r"\\end{tabular}"
_latex_template = r"""
    \documentclass{{article}}

    \usepackage{{multirow}}
    \usepackage{{xeCJK}}
    \setCJKmainfont[Path = ./,
        Extension = .otf,]{{NotoSerifCJKtc-Black.otf}}

    \begin{{document}}

    \hspace{{-5cm}}
    {latex_table_str}

    \end{{document}}"""

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))


def convert_latex_table_to_pandas(
    latex_table_str: str,
    headers: Union[bool, Sequence[str], None] = None,
) -> pd.DataFrame:
    process_latex_table_str = re.sub(_latex_table_begin_pattern, "", latex_table_str)
    process_latex_table_str = re.sub(_latex_table_end_pattern, "", process_latex_table_str)
    rows = process_latex_table_str.replace("\n", " ").strip().split(r"\\")
    rows = [row.strip() for row in rows if "&" in row]  # 過濾掉無關的行

    # 拆分表格各儲存格
    table_data = [row.replace("\\\\", "").replace(r"\hline", "").strip().split("&") for row in rows]
    cleaned_data = [[cell.strip() for cell in row] for row in table_data]

    if isinstance(headers, bool) and headers:
        headers = cleaned_data[0]  # 第一行是列名
        data = cleaned_data[1:]  # 剩餘的是數據
        df = pd.DataFrame(data, columns=headers)
    elif headers:
        df = pd.DataFrame(cleaned_data)
    return df


def merge_horizontal_cell(
    latex_table_str: str,
    rng: random.Random = None,
    count: int = 1,
    content: str = None,
    **kwds,
) -> Tuple[str, str]:
    logger.debug("Run merge_horizontal_cell")
    result = re.findall(_latex_table_begin_pattern, latex_table_str)
    if not result:
        raise ValueError("Not latex table")
    elif "multicolumn" in latex_table_str:
        raise ValueError("Not support convert have multicolumn latex")

    begin_str = result[0]
    end_str = r"\end{tabular}"
    process_latex_table_str = re.sub(_latex_table_begin_pattern, "", latex_table_str)
    process_latex_table_str = re.sub(_latex_table_end_pattern, "", process_latex_table_str)

    rows_multicolumn = [s.strip() for s in process_latex_table_str.split(r"\\") if s.strip()]
    rows_repeat = [s.strip() for s in process_latex_table_str.split(r"\\") if s.strip()]
    rand_nums = [i for i in range(1, len(rows_multicolumn))]
    rng.shuffle(rand_nums)
    rand_nums = rand_nums[:count]

    for rand_num in rand_nums:
        texts = rows_multicolumn[rand_num].replace(r"\hline", "").strip().split("&")
        texts_str = "".join(texts) if not content else content
        texts_str_repeat = "&".join([texts_str for _ in range(len(texts))])
        rows_multicolumn[rand_num] = rf"\hline \multicolumn{{{len(texts)}}}{{|c|}}{{{texts_str}}}"
        rows_repeat[rand_num] = f"\hline {texts_str_repeat}"

    latex_table_multicolumn_str = r"\\".join(rows_multicolumn)
    latex_table_repeat_str = r"\\".join(rows_repeat)
    return (
        f"{begin_str}\n{latex_table_multicolumn_str}\n{end_str}",
        f"{begin_str}\n{latex_table_repeat_str}\n{end_str}",
    )


def merge_vertical_cell(
    latex_table_str: str,
    rng: random.Random = None,
    count: int = 1,
    content: str = None,
    vertical: Tuple[int, Tuple[int, int]] = 1,
    **kwds,
) -> Tuple[str, str]:
    logger.debug("Run merge_vertical_cell")
    if count != 1:
        raise NotImplementedError("Not support count > 1")

    result = re.findall(_latex_table_begin_pattern, latex_table_str)
    if not result:
        raise ValueError("Not latex table")
    elif "multicolumn" in latex_table_str:
        raise ValueError("Not support convert have multicolumn latex")

    begin_str = result[0]
    end_str = r"\end{tabular}"

    table = convert_latex_table_to_pandas(latex_table_str, headers=True)
    rand_nums = [i for i in range(len(table) - 1)]
    rng.shuffle(rand_nums)
    rand_nums = rand_nums[:count]
    logger.debug(f"rand_nums: {rand_nums}")

    rows = [
        r"\hline " + " & ".join([str(c) for c in table.columns]),
    ]
    for rand_num in rand_nums:
        added_multirow = False
        start_add_cline = False
        col = rng.randint(0, len(table.columns) - 1)
        logger.debug(f"col: {col}")
        if isinstance(vertical, int):
            multirow_num = vertical + 1
        elif isinstance(vertical, (tuple, list)) and len(vertical) >= 2:
            multirow_num = rng.randint(vertical[0], min(vertical[1], len(table) - rand_num)) + 1
        else:
            raise TypeError(f"vertical should be int or tuple{vertical}")

        for i in range(len(table)):
            contents = list()
            if i in [rand_num + n for n in range(multirow_num)]:  # add multirow and cline
                for j, v in enumerate(table.iloc[i]):
                    if j == col:
                        if not added_multirow:
                            contents.append(rf"\multirow{{{multirow_num}}}{{*}}{{{content}}}")
                            added_multirow = True
                        else:
                            contents.append("")
                    else:
                        contents.append(v)

                if start_add_cline:
                    if col == 0:
                        contents[0] = rf"\cline{{{col+2}-{len(table.columns)}}} {contents[0]}"
                    elif col == len(table.columns) - 1:
                        contents[0] = rf"\cline{{1-{len(table.columns) - 1}}} {contents[0]}"
                    else:
                        contents[0] = rf"\cline{{1-{col}}} \cline{{{col+2}-{len(table.columns)}}} {contents[0]}"
                start_add_cline = True
            else:
                contents = [str(e) for e in table.iloc[i]]
            contents = " & ".join(contents)
            rows.append(rf"\hline {contents}" if r"\cline" not in contents else contents)

    rows.append(r"\hline")
    final_latex_table_str = r"\\".join(rows)
    return (
        f"{begin_str}\n{final_latex_table_str}\n{end_str}",
        f"{begin_str}\n{final_latex_table_str}\n{end_str}",
    )


def latex_table_to_image(
    latex_table_str: str,
    dpi: int = 300,
    timeout: int = 10,
) -> PILImage.Image:
    with TemporaryDirectory(prefix="latex_temp") as temp_dir:
        # 定義文件路徑
        tex_file = Path(temp_dir, "formula.tex")
        pdf_file = Path(temp_dir, "formula.pdf")

        # 寫入 LaTeX 文件
        tex_file.write_text(
            _latex_template.format(
                latex_table_str=latex_table_str,
            ),
            encoding="utf-8",
        )

        try:
            # 編譯 LaTeX 文件為 PDF
            try:
                subprocess.run(
                    ["xelatex", "-output-directory", str(temp_dir), str(tex_file)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                )
            except subprocess.CalledProcessError as e:
                logger.error("Error in LaTeX compilation:", e.stderr.decode("utf-8"))
                return
            except subprocess.TimeoutExpired as e:
                logger.error("ERROR: Convert latex table to pdf failed")
                return

            try:
                images = convert_from_path(pdf_file, dpi=dpi, fmt="png")
                if isinstance(images, list) and images:
                    return images[0]

            except Exception as e:
                logger.exception(e)
        except Exception:
            pass
    return None


if __name__ == "__main__":
    rng = random.Random(os.environ.get("SEED", None))
    latex_table_str = r"""\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|}
    \hline 編號 & 組編號 & 號數 & 長A & 型狀/長度B & 長C & 總長度 & 支數 & 重量 & 備註 \\
    \hline 1 & 彎料 & \#5 & & 450 & & 480 & 10 & 75 & \\
    \hline 2 & 彎料 & \#4 & 80 & 80 & & 340 & 720 & 2433 & \\
    \hline 3 & 彎料 & \#4 & 65 & 80 & & 310 & 10 & 31 & \\
    \hline 4 & 彎料 & \#4 & 10 & 81 & 12 & 105 & 2800 & 2922 & \\
    \hline
    \end{tabular}"""

    latex_table_str_ = r"""\begin{tabular}{|c|c|c|}
\hline
\multirow{3}{*}{合併的儲存格} & \multicolumn{2}{|c|}{跨兩列的儲存格} \\
\cline{2-3} & 第一列 & 第二列 \\
\cline{2-3} & 第一列a & 第二列b \\
\hline
\end{tabular}"""

    latex_table_image_str, latex_table_label_str = merge_vertical_cell(latex_table_str, rng=rng, content="以下空白")
    logger.debug(latex_table_image_str)

    Path("./outputs").mkdir(exist_ok=True)

    image = latex_table_to_image(
        latex_table_image_str,
    )
    if image:
        image.save("outputs/origin-output.png")
        tables = run_table_detect(image)
        crop_table_images = crop_table_bbox(src=image, tables=tables, margin=10)
        if crop_table_images:
            plt.imsave("outputs/output.png", crop_table_images[0])

    image = latex_table_to_image(
        latex_table_label_str,
    )
    if image:
        image.save("outputs/origin-label.png")
        tables = run_table_detect(image)
        crop_table_images = crop_table_bbox(src=image, tables=tables, margin=10)
        if crop_table_images:
            plt.imsave("outputs/label.png", crop_table_images[0])
