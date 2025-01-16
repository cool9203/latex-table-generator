# coding: utf-8

import argparse
import logging
import os
import pprint
from pathlib import Path

from PIL import Image as PILImage

image_extensions = {ex for ex, f in PILImage.registered_extensions().items() if f in PILImage.OPEN}

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
    parser.add_argument("-s", "--seed", type=str, default=None, help="Random seed")
    parser.add_argument("--specific_headers", type=str, nargs="+", default=[], help="Choice column name to merge")
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

    parser.add_argument("--tqdm", action="store_true", help="Use tqdm to show progress bar")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = arg_parser()

    check_arguments = [
        "seed",
        "h_contents",
        "v_contents",
        "vh_contents",
        "specific_headers",
    ]
    range_arguments = [
        "vertical",
        "horizontal_count",
        "vertical_count",
        "skew_angle",
        "new_image_size",
        "rows_range",
    ]

    # Pre-process arguments
    for check_argument in check_arguments:
        if not getattr(args, check_argument):
            delattr(args, check_argument)

    # Process range argument
    for range_argument in range_arguments:
        arg = getattr(args, range_argument)
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

    # Process image_path argument
    image_paths = list()
    for image_folder_path in args.image_paths:
        path = Path(image_folder_path)
        if path.is_dir() and path.exists():
            for image_path in path.iterdir():
                if image_path.suffix.lower() in image_extensions:
                    image_paths.append(str(image_path.resolve()))
            logger.debug(f"Load '{path!s}' image data path")
        elif path.is_file() and path.exists():
            if path.suffix.lower() in image_extensions:
                image_paths.append(str(path.resolve()))
            logger.debug(f"Load '{path!s}' image")
        else:
            logger.debug(f"Not Load '{image_folder_path}' image data path, path not exists")
    args.image_paths = image_paths

    args = vars(args)
    print(pprint.pformat(args))

    from latex_table_generator.main import get_subfolder_path, main

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
