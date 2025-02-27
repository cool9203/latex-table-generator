# coding: utf-8

import ast
import json
import logging
import os
import re
from io import StringIO
from os import PathLike
from pathlib import Path
from typing import Any, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from latex_table_generator.errors import (
    FormatError,
    NotHtmlError,
    NotLatexError,
    NotSupportLatexError,
    NotSupportMultiLatexTableError,
)

_latex_table_begin_pattern = r"\\begin{tabular}{[lrc|]*}"
_latex_table_end_pattern = r"\\end{tabular}"
_latex_multicolumn_pattern = r"\\multicolumn{(\d+)}{([lrc|]+)}{(.*)}"
_latex_multirow_pattern = r"\\multirow{(\d+)}{([\*\d]+)}{(.*)}"
_html_table_begin_pattern = r"<table>[\s\w]*(?:<thead>)?"
_html_table_end_pattern = r"</table>"


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))


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


def check_overlap(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> bool:
    """Algorithm reference from: https://stackoverflow.com/a/306332

    Args:
        a (Tuple[int, int, int, int]): (x1, y1, x2, y2)
        b (Tuple[int, int, int, int]): (x1, y1, x2, y2)

    Returns:
        bool: two image is overlap
    """
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _remove_all_space_row(
    rows: list[list[Any]],
) -> list[list[str]]:
    cleaned_rows = list()
    for row in rows:
        _row_data = list()
        for cell in row:
            _row_data.append(str(cell).strip())
        if len("".join(_row_data)) == 0:
            continue
        else:
            cleaned_rows.append(_row_data)
    return cleaned_rows


def preprocess_latex_table_string(
    latex_table_str: str,
) -> str:
    processed_latex_table_str = re.sub(_latex_table_begin_pattern, "", latex_table_str)
    processed_latex_table_str = re.sub(_latex_table_end_pattern, "", processed_latex_table_str)
    processed_latex_table_str = processed_latex_table_str.replace("\n", " ").strip()

    # Fix multiple \hline and \hline not at start of row error
    rows = processed_latex_table_str.split(r"\\")
    new_rows = list()
    for row in rows:
        _row = row
        if row.count(r"\hline") > 0:
            _row = _row.replace(r"\hline", "").strip()
            _row = rf"\hline {_row}"
        new_rows.append(_row)

    return "\\\\\n".join(new_rows)


def is_latex_table(table_str: str):
    return len(re.findall(_latex_table_begin_pattern, table_str)) >= 1


def is_html_table(table_str: str):
    return len(re.findall(_html_table_begin_pattern, table_str)) >= 1


def pre_check_latex_table_string(
    latex_table_str: str,
) -> Tuple[str, str]:
    results = re.findall(_latex_table_begin_pattern, latex_table_str)
    if not results:
        raise NotLatexError("Not latex table")
    elif len(results) > 1:
        raise NotSupportMultiLatexTableError("Not support convert have multi latex table")

    begin_str = results[0]
    end_str = r"\end{tabular}"
    return (begin_str, end_str)


def convert_latex_table_to_pandas(
    latex_table_str: str,
    headers: Union[bool, Sequence[str], None] = None,
    unsqueeze: bool = False,
    remove_all_space_row: bool = False,
    **kwds,
) -> pd.DataFrame:
    pre_check_latex_table_string(latex_table_str=latex_table_str)
    processed_latex_table_str = preprocess_latex_table_string(latex_table_str)
    rows = [
        row.replace("\n", "").strip()
        for row in processed_latex_table_str.split(r"\\")
        if ("&" in row or r"\multicolumn" in row) and row.replace("\n", "").strip()
    ]  # Filter unrelated row data

    # Split latex table to list table
    cleaned_data = list()
    table_data = [row.replace(r"\\", "").replace(r"\hline", "").replace(r"\cline", "").strip().split("&") for row in rows]
    for row in table_data:
        _row_data = list()
        for cell_text in row:
            _cell_text = cell_text.strip()
            if re.match(_latex_multicolumn_pattern, _cell_text):
                multicolumn_data = re.findall(_latex_multicolumn_pattern, _cell_text)[0]
                for index in range(int(multicolumn_data[0])):
                    if unsqueeze:
                        _row_data.append(multicolumn_data[2].strip())
                    else:
                        if index == 0:
                            _row_data.append(
                                rf"\multicolumn{{{multicolumn_data[0]}}}{{{multicolumn_data[1]}}}{{{multicolumn_data[2].strip()}}}"
                            )
                        else:
                            _row_data.append("")
            else:
                _row_data.append(_cell_text)
        cleaned_data.append(_row_data)

    # Process multirow
    for col in range(len(cleaned_data)):
        for row in range(len(cleaned_data[col])):
            # Clean multi row data
            multirow_result = re.findall(_latex_multirow_pattern, cleaned_data[col][row])
            if multirow_result:
                if unsqueeze:
                    for offset in range(int(multirow_result[0][0])):
                        cleaned_data[col + offset][row] = multirow_result[0][2].strip()
                else:
                    cleaned_data[col][row] = (
                        rf"\multirow{{{multirow_result[0][0]}}}{{{multirow_result[0][1]}}}{{{multirow_result[0][2].strip()}}}"
                    )
                    for offset in range(1, int(multirow_result[0][0])):
                        if col + offset >= len(cleaned_data):
                            break
                        cleaned_data[col + offset][row] = ""

    if remove_all_space_row:
        cleaned_data = _remove_all_space_row(rows=cleaned_data)

    try:
        if headers:
            if isinstance(headers, bool):
                headers = cleaned_data[0]  # First row is header
                cleaned_data = cleaned_data[1:]  # Other row is row data

            # Filling every row length to headers length
            for i in range(len(cleaned_data)):
                if len(cleaned_data[i]) > len(headers):
                    cleaned_data[i] = cleaned_data[i][: len(headers)]
                elif len(cleaned_data[i]) < len(headers):
                    cleaned_data[i] += ["" for _ in range(len(headers) - len(cleaned_data[i]))]
            df = pd.DataFrame(cleaned_data, columns=headers)
        else:
            df = pd.DataFrame(cleaned_data)
    except ValueError as e:
        raise NotSupportLatexError("Not support this latex") from e

    return df


def convert_pandas_to_latex(
    df: pd.DataFrame,
    full_border: bool = False,
) -> str:
    _row_before_text = ""
    if full_border:
        _row_before_text = r"\hline "
        latex_table_str = f"\\begin{{tabular}}{{{'c'.join(['|' for _ in range(len(df.columns) + 1)])}}}\n"
    else:
        latex_table_str = f"\\begin{{tabular}}{{{''.join(['c' for _ in range(len(df.columns))])}}}\n"

    # Add header
    latex_table_str += _row_before_text + f"{'&'.join([column for column in df.columns])}\\\\\n"

    # Add row data
    for i in range(len(df)):
        row = list()
        skip_count = 0
        for column_index in range(len(df.columns)):
            if skip_count > 0:
                skip_count -= 1
            else:
                multicolumn_result = re.findall(_latex_multicolumn_pattern, df.iloc[i, column_index])
                skip_count = int(multicolumn_result[0][0]) - 1 if multicolumn_result and skip_count == 0 else skip_count
                row.append(df.iloc[i, column_index])
        latex_table_str += _row_before_text + f"{'&'.join(row)}\\\\\n"

    if full_border:
        latex_table_str += "\\hline\n"
    latex_table_str += r"\end{tabular}"

    return latex_table_str


def convert_html_table_to_pandas(
    html_table_str: str,
    remove_all_space_row: bool = False,
    **kwds,
) -> pd.DataFrame:
    try:
        with StringIO(html_table_str) as f:
            dfs = pd.read_html(
                io=f,
                keep_default_na=False,
            )

        if remove_all_space_row:
            return [
                pd.DataFrame(
                    _remove_all_space_row(rows=df.values.tolist()),
                    columns=df.columns,
                )
                for df in dfs
            ][0]
        else:
            return dfs[0]
    except Exception:
        raise NotHtmlError("This table str not is html")


def convert_table_to_pandas(
    table_str: str,
    headers: Union[bool, Sequence[str], None] = None,
    unsqueeze: bool = False,
    remove_all_space_row: bool = False,
    **kwds,
) -> pd.DataFrame:
    if is_latex_table(table_str):
        return convert_latex_table_to_pandas(
            latex_table_str=table_str,
            headers=headers,
            unsqueeze=unsqueeze,
            remove_all_space_row=remove_all_space_row,
            **kwds,
        )
    elif is_html_table(table_str):
        return convert_html_table_to_pandas(
            html_table_str=table_str,
            unsqueeze=unsqueeze,
            remove_all_space_row=remove_all_space_row,
            **kwds,
        )
    else:
        raise FormatError("Not Support convert the format table")


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
