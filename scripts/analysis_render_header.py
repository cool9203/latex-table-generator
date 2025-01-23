# coding: utf-8

import dataclasses
import logging
import math
import os
import pprint
import re
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import pandas as pd
import tqdm as TQDM

from latex_table_generator.errors import LatexTableGeneratorError
from latex_table_generator.utils import convert_latex_table_to_pandas, get_subfolder_path

_number_pattern = re.compile(r"^(?:\\?#(\d)+)|(?:(\d)+\\?#)|(\d+)")  # Have process hashtag

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(os.environ.get("LOG_LEVEL", "WARNING"))


@dataclasses.dataclass
class RenderHeaderRole:
    names: List[str]
    type: str
    hashtag: bool
    sequence: bool
    range: Sequence[int] = dataclasses.field(default_factory=[])
    choices: Sequence[str] = dataclasses.field(default_factory=[])


def update_dict(
    old: List[Dict[str, Any]],
    new: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    assert len(old) == len(new), "Can't not update not same length's List[Dict]"

    final = list()
    for i in range(len(old)):
        is_number = old[i]["type"] in ["int", "float"]
        final.append(
            {
                "names": list(set(old[i]["names"] + new[i]["names"])),
                "type": old[i]["type"],
                "hashtag": old[i]["hashtag"],
                "sequence": old[i]["sequence"],
                "range": [min(old[i]["range"][0], new[i]["range"][0]), max(old[i]["range"][1], new[i]["range"][1])]
                if is_number
                else None,
                "choices": list(set(old[i]["choices"] + new[i]["choices"])) if not is_number else None,
            }
        )
    return final


def _check_number_and_hashtag(
    number_value: str,
) -> Tuple[Union[str, int, float], bool]:
    _number_value = number_value.replace(",", "").replace(" ", "")
    results = re.findall(_number_pattern, _number_value)
    for result in results:
        for n in result:
            _number_value = n if n else _number_value

    # Check is number and need hash tag
    value = number_value
    try:
        value = int(_number_value)
    except ValueError:
        try:
            value = float(_number_value)
        except ValueError:
            pass
    return (value, "#" in number_value)


def _analysis_render_header(
    df: pd.DataFrame,
) -> Tuple[str, List[Dict[str, Any]]]:
    # Get space row index
    space_row_indexes = list()
    for i in range(len(df)):
        is_space_row = True
        for cell in df.iloc[i]:
            if str(cell):
                is_space_row = False
                break
        if is_space_row:
            space_row_indexes.append(i)
    if len(space_row_indexes) == len(df):
        return (None, None)

    logger.debug(f"space_row_indexes: {space_row_indexes}")
    headers = list()
    for column_name in df.columns:
        is_int = None
        is_float = None
        number_range = [math.inf, -math.inf]
        sequence = None
        hashtag = None
        choices = list()
        start_number = None

        for i in range(len(df)):
            if i not in space_row_indexes:
                value = str(df.iloc[i][column_name])
                (value, hashtag) = _check_number_and_hashtag(value)
                (is_int, is_float) = (isinstance(value, int), isinstance(value, float))
                start_number = value if is_int and start_number is None else start_number
                if isinstance(value, str):
                    if "dtype" in value.lower():
                        raise ValueError(f"value error: {value}")
                    elif value and value not in choices:
                        choices.append(value)

                # Update number range
                if is_int | is_float:
                    number_range = [min(value, number_range[0]), max(value, number_range[1])]

                # Check sequence
                if is_int and i > 0:
                    if sequence is not False and value == start_number + i:
                        sequence = True
                    else:
                        sequence = False

        headers.append(
            {
                "names": [column_name],
                "type": "int" if is_int else ("float" if is_float else "str"),
                "hashtag": hashtag,
                "sequence": sequence if (is_int | is_float) else False,
                "range": number_range if (is_int | is_float) else None,
                "choices": choices if not (is_int | is_float) else None,
            }
        )

    ids = list()
    all_choice_is_empty = True
    for header in headers:
        if all_choice_is_empty and header.get("choices"):
            all_choice_is_empty = False

        _ids = list()
        if header["hashtag"]:
            _ids.append("h")
        if header["sequence"]:
            _ids.append("s")
        _ids.append(header["type"])
        ids.append("".join(_ids))

    if all_choice_is_empty:
        return (None, headers)
    return ("-".join(ids), headers)


def analysis_render_header(
    root_paths: List[PathLike],
    output_path: PathLike = "render_headers",
    tqdm: bool = False,
):
    logger.debug("Run analysis_render_header")
    filepaths = list()
    for root_path in TQDM.tqdm(root_paths, desc="Search analyzable file") if tqdm else root_paths:
        logger.debug(f"Search {root_path}")
        filepaths += get_subfolder_path(
            path=root_path,
            targets=(
                ("*.jpg", "*.png", "*.JPG", "*.PNG"),
                ("*.txt",),
            ),
        )

    analysis_results = dict()
    for filepath in TQDM.tqdm(filepaths, desc="Analysis file") if tqdm else filepaths:
        for file in Path(filepath).glob("*.txt"):
            try:
                with file.open("r", encoding="utf-8") as f:
                    df = convert_latex_table_to_pandas(f.read(), headers=True)
                (key, result) = _analysis_render_header(df=df)
                if key is None:
                    logger.info(f"Skip empty table, file: {file!s}")
                    continue
                logger.debug(pprint.pformat(result))
                if key in analysis_results:
                    analysis_results[key] = update_dict(
                        old=analysis_results[key],
                        new=result,
                    )
                else:
                    analysis_results[key] = result
                logger.info(f"'{file!s}' -> '{key!s}'")
                logger.debug(pprint.pformat(analysis_results))
                logger.debug("-" * 25)
            except (LatexTableGeneratorError, ValueError) as e:
                logger.error(f"Skip file '{file!s}' {e!s}")

    for name, analysis_result in analysis_results.items():
        column_length = len(name.split("-"))
        with Path(output_path, f"{column_length}-{name}.py").open("w", encoding="utf-8") as f:
            f.write(pprint.pformat(analysis_result))


if __name__ == "__main__":
    analysis_render_header(
        root_paths=[
            "/mnt/c/Users/ychsu/Downloads/沛波標記data/要標記資料/20241021_需Label鋼材data/3.工具框選表格_第四次提供(100)",
            "/mnt/c/Users/ychsu/Downloads/沛波標記data/要標記資料/20241104_需Label鋼材data",
            "/mnt/c/Users/ychsu/Downloads/沛波標記data/要標記資料/20241108_需Label鋼材data",
            "/mnt/c/Users/ychsu/Downloads/沛波標記data/要標記資料/20241203_需Label鋼材data",
            "/mnt/c/Users/ychsu/Downloads/沛波標記data/要標記資料/20241205_需Label鋼材data",
            "/mnt/c/Users/ychsu/Downloads/沛波標記data/要標記資料/20241213_需Label鋼材Data/單一正規",
            "/mnt/c/Users/ychsu/Downloads/沛波標記data/要標記資料/20241216_需Label鋼材Data/1_單一正規",
            "/mnt/c/Users/ychsu/Downloads/沛波標記data/要標記資料/20241218_需Label鋼材Data/單一正規",
            "/mnt/c/Users/ychsu/Downloads/沛波標記data/要標記資料/20241223_需Label鋼材Data/1_單正規表格",
            "/mnt/c/Users/ychsu/Downloads/沛波標記data/要標記資料/20241225_需Label鋼材Data/1_單正規表格",
        ],
    )
