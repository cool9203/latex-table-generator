# coding: utf-8

import argparse
from io import StringIO
from os import PathLike
from pathlib import Path
from typing import (
    List,
    Sequence,
    Union,
)

import numpy as np
import pandas as pd
import pypandoc
import tqdm as TQDM

from latex_table_generator.utils import get_subfolder_path


def arg_parser() -> argparse.Namespace:
    """取得執行程式時傳遞的參數

    tutorial: https://docs.python.org/zh-tw/3/howto/argparse.html#
    reference: https://docs.python.org/zh-tw/3/library/argparse.html#nargs

    Returns:
        argparse.Namespace: 使用args.name取得傳遞的參數
    """

    parser = argparse.ArgumentParser(description="Convert latex table to markdown")

    parser.add_argument("-p", "--paths", type=str, nargs="+", required=True, help="Input data path, can be folder")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output path")
    parser.add_argument("--tqdm", action="store_true", help="Use tqdm to show progress bar")
    parser.add_argument("-f", "--format", type=str, help="Read table data's format, like 'latex', 'markdown', 'html'")
    parser.add_argument("-t", "--to", type=str, help="Convert table data's format, like 'latex', 'markdown', 'html'")
    parser.add_argument("-e", "--extensions", type=str, nargs="+", default=[".txt"], help="Read table data's extensions")

    args = parser.parse_args()

    return args


def convert_table(
    src: str,
    format: str,
    to: str,
) -> List[str]:
    if format == "html":
        html = src
    else:
        html = pypandoc.convert_text(source=src, to="html", format=format)
    with StringIO(html) as f:
        dfs = pd.read_html(f)

    if hasattr(dfs[0], f"to_{to}"):
        return [getattr(df, f"to_{to}")(index=False).replace("nan", "   ") for df in dfs]
    else:
        raise ValueError(f"Not support convert to '{to}'")


if __name__ == "__main__":
    args = arg_parser()

    paths = TQDM.tqdm(args.paths) if args.tqdm else args.paths
    Path(args.output).mkdir(parents=True, exist_ok=True)

    extensions = args.extensions

    for path in paths:
        subfolder_paths = get_subfolder_path(
            path,
            targets=(("*.txt"),),
        )
        for subfolder_path in subfolder_paths:
            filenames = list()
            for extension in extensions:
                filenames += [f for f in subfolder_path.glob(rf"*{extension}")]
            filenames = TQDM.tqdm(filenames, leave=False) if args.tqdm else filenames
            for filename in filenames:
                with Path(filename).open("r", encoding="utf-8") as f:
                    markdown_table_strs = convert_table(f.read(), format=args.format, to=args.to)
                with Path(args.output, filename.stem + ".txt").open("w", encoding="utf-8") as f:
                    f.write("\n\n".join(markdown_table_strs))
