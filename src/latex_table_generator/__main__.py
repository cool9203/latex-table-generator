# coding: utf-8

import argparse
import logging
import os
import pprint
from pathlib import Path

from latex_table_generator.utils import load_render_header_file

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))


def arg_parser() -> argparse.Namespace:
    """取得執行程式時傳遞的參數

    tutorial: https://docs.python.org/zh-tw/3/howto/argparse.html#
    reference: https://docs.python.org/zh-tw/3/library/argparse.html#nargs

    Returns:
        argparse.Namespace: 使用args.name取得傳遞的參數
    """

    parser = argparse.ArgumentParser(description="Auto generate latex table data")

    parser.add_argument("-o", "--output_path", required=True, help="Output path")
    parser.add_argument("-i", "--input_path", type=str, nargs="+", default=[], help="Input path(folder)")
    parser.add_argument("-c", "--count", type=int, default=None, help="Full random generate latex table")
    parser.add_argument(
        "-m",
        "--merge_methods",
        type=str,
        nargs="+",
        choices=["vertical", "horizontal", "hybrid", "none"],
        default=["vertical", "horizontal", "hybrid", "none"],
        help="Merge methods",
    )
    parser.add_argument(
        "-vc", "--v_contents", type=str, nargs="+", default=[], help="Merged vertical cell content, will random choice"
    )
    parser.add_argument(
        "-hc", "--h_contents", type=str, nargs="+", default=[], help="Merged horizontal cell content, will random choice"
    )
    parser.add_argument(
        "-vhc",
        "--vh_contents",
        type=str,
        nargs="+",
        default=[],
        help="Merged horizontal and vertical cell content, will random choice",
    )
    parser.add_argument("--render_headers", type=str, nargs="+", default=None, help="Render header setting")
    parser.add_argument("-s", "--seed", type=str, default=None, help="Random seed")
    parser.add_argument("--log_level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], default=None, help="Log level")
    parser.add_argument("--specific_headers", type=str, nargs="+", default=[], help="Choice column name to merge")
    parser.add_argument("--image_specific_headers", type=str, nargs="+", default=[], help="Choice column name to merge")
    parser.add_argument("--vertical", type=str, nargs="+", default=[], help="Vertical length can be range")
    parser.add_argument("--image_paths", type=str, nargs="+", default=[], help="Paste to table cell image data path")
    parser.add_argument(
        "-h_count", "--horizontal_count", type=int, nargs="+", default=[1, 3], help="Merge horizontal Run times in a image"
    )
    parser.add_argument(
        "-v_count", "--vertical_count", type=int, nargs="+", default=[1, 3], help="Merge vertical Run times in a image"
    )
    parser.add_argument("--skew_angle", type=int, nargs="+", default=[-5, 5], help="Table image rotate angle")
    parser.add_argument("--new_image_size", type=int, nargs="+", default=[2480, 3508], help="Full random mode new image size")
    parser.add_argument(
        "--min_crop_size",
        type=float,
        default=None,
        help="Full random mode min table area size, support 0~1 to auto get image size percentage",
    )
    parser.add_argument("--rows_range", type=int, nargs="+", default=[1, 20], help="Full random mode table rows count range")
    parser.add_argument(
        "--format",
        type=str,
        nargs="+",
        choices=["latex", "markdown", "html", "table_info", "all"],
        default=["all"],
        help="Output label format",
    )
    parser.add_argument("--multi_table", type=int, default=None, help="Multi table number")
    parser.add_argument(
        "--multi_table_paste_vertical",
        type=str,
        choices=["random", "always", "none"],
        default="none",
        help="Multi table pasting vertical",
    )
    parser.add_argument("--html_label_cell_merge", action="store_true", help="Html label will output merge cell format")
    parser.add_argument(
        "--add_space_row_percentage",
        type=float,
        default=0.3,
        help="Full random mode add space row percentage",
    )
    parser.add_argument(
        "--dropout_percentage",
        type=float,
        default=None,
        help="Dropout table content percentage",
    )

    parser.add_argument("--tqdm", action="store_true", help="Use tqdm to show progress bar")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = arg_parser()

    check_arguments = [
        "seed",
        "log_level",
        "h_contents",
        "v_contents",
        "vh_contents",
        "specific_headers",
        "image_specific_headers",
        "render_headers",
    ]
    range_arguments = [
        "vertical",
        "horizontal_count",
        "vertical_count",
        "skew_angle",
        "new_image_size",
        "rows_range",
    ]
    file_path_arguments = [
        "render_headers",
    ]
    split_txt_file_path_arguments = [
        "h_contents",
        "v_contents",
        "vh_contents",
    ]

    # Pre-process arguments
    for check_argument in check_arguments:
        if hasattr(args, check_argument) and not getattr(args, check_argument):
            delattr(args, check_argument)

    # Process range argument
    for range_argument in range_arguments:
        arg = getattr(args, range_argument, [])
        if len(arg) > 1:
            setattr(
                args,
                range_argument,
                [int(arg[0]), int(arg[1])] if int(arg[0]) < int(arg[1]) else [int(arg[1]), int(arg[0])],
            )
        elif len(arg) == 1:
            setattr(args, range_argument, int(arg[0]))
        else:
            delattr(args, range_argument)

    # Process file path argument
    for file_path_argument in file_path_arguments:
        arg = getattr(args, file_path_argument, None)
        if arg is not None:
            if isinstance(arg, str):
                with Path(arg).open("r", encoding="utf-8") as f:
                    setattr(args, file_path_argument, f.read())
            else:
                contents = list()
                for arg_element in arg:
                    path = Path(arg_element)
                    if not path.exists():  # Normal string
                        contents.append(arg_element)
                    elif path.is_file():  # File path
                        contents.append(load_render_header_file(path=path))
                    else:  # Folder path
                        for _path in path.iterdir():
                            contents.append(load_render_header_file(path=_path))
                setattr(args, file_path_argument, contents)
        else:
            delattr(args, file_path_argument)

    # Process split txt file path argument
    for split_txt_file_path_argument in split_txt_file_path_arguments:
        arg = getattr(args, split_txt_file_path_argument, None)
        if arg is not None:
            contents = list()
            for content in arg:
                if Path(content).exists() and Path(content).is_file():
                    with Path(content).open("r", encoding="utf-8") as f:
                        contents += f.read().split("\n")
                else:
                    contents.append(content)
            setattr(args, split_txt_file_path_argument, contents)
        else:
            delattr(args, file_path_argument)

    args = vars(args)
    print(pprint.pformat(args))

    from latex_table_generator.main import main
    from latex_table_generator.utils import get_subfolder_path

    input_paths = args.pop("input_path")
    if input_paths:
        for input_path in input_paths:
            subfolder_paths = get_subfolder_path(input_path)
            logger.debug(f"subfolder_paths: {subfolder_paths}")
            for subfolder_path in subfolder_paths:
                main(
                    input_path=subfolder_path,
                    **args,
                )
    else:
        main(**args)
