# coding: utf-8

import argparse
import logging
import os
import pprint
import random
from os import PathLike
from pathlib import Path
from typing import List, Tuple, Union

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

    parser.add_argument("-i", "--input_path", required=True, help="Input path(folder)")
    parser.add_argument("-o", "--output_path", required=True, help="Output path")
    parser.add_argument(
        "-m", "--merge_method", type=str, choices=["random", "vertical", "horizontal"], default="random", help="Merge method"
    )
    parser.add_argument("-vc", "--v_contents", type=str, nargs="+", default=[], help="Merged cell content, will random choice")
    parser.add_argument("-hc", "--h_contents", type=str, nargs="+", default=[], help="Merged cell content, will random choice")
    parser.add_argument("-s", "--seed", type=str, default=None, help="Random seed")
    parser.add_argument("--specific_headers", type=str, nargs="+", default=[], help="Choice column name to merge")
    parser.add_argument("--vertical", type=str, nargs="+", default=[], help="Vertical length can be range")

    args = parser.parse_args()

    return args


def main(
    input_path: PathLike,
    output_path: PathLike,
    merge_method: str = "random",
    h_contents: List[str] = ["開口補強"],
    v_contents: List[str] = ["彎鉤", "鋼材筋"],
    specific_headers: List[str] = [".*備註.*"],
    vertical: Union[int, Tuple[int, int]] = [1, 5],
    **kwds,
):
    from matplotlib import pyplot as plt

    from latex_table_generator.base import draw_table_bbox, paste_image_with_table_bbox
    from latex_table_generator.main import (
        PILImage,
        convert_latex_table_to_pandas,
        get_fit_size_latex_table_to_image,
        merge_horizontal_cell,
        merge_vertical_cell,
        run_table_detect,
    )

    rng = random.Random(kwds.get("seed", os.environ.get("SEED", None)))

    for filename in Path(input_path).glob(r"*.txt"):
        if Path(input_path, f"{filename.stem}.jpg").exists():
            file_image = PILImage.open(Path(input_path, f"{filename.stem}.jpg"))
        elif Path(input_path, f"{filename.stem}.png").exists():
            file_image = PILImage.open(Path(input_path, f"{filename.stem}.png"))
        else:
            logger.info(f"Not found {filename}.jpg or {filename}.png, skip file")
            continue

        logger.info(f"Run {filename.name}")

        with filename.open("r", encoding="utf-8") as f:
            latex_table_str = f.read()

        try:
            if merge_method == "random":
                _rand_num = rng.randint(0, 1)
                if _rand_num == 0:
                    rand_content_index = rng.randint(0, len(h_contents) - 1)
                    latex_table_image_str, latex_table_label_str = merge_horizontal_cell(
                        latex_table_str=latex_table_str,
                        rng=rng,
                        content=h_contents[rand_content_index],
                    )
                else:
                    rand_content_index = rng.randint(0, len(v_contents) - 1)
                    latex_table_image_str, latex_table_label_str = merge_vertical_cell(
                        latex_table_str=latex_table_str,
                        rng=rng,
                        content=v_contents[rand_content_index],
                        specific_headers=specific_headers,
                        vertical=vertical,
                    )
            elif merge_method == "vertical":
                rand_content_index = rng.randint(0, len(v_contents) - 1)
                latex_table_image_str, latex_table_label_str = merge_vertical_cell(
                    latex_table_str=latex_table_str,
                    rng=rng,
                    content=v_contents[rand_content_index],
                    specific_headers=specific_headers,
                    vertical=vertical,
                )
            elif merge_method == "horizontal":
                rand_content_index = rng.randint(0, len(h_contents) - 1)
                latex_table_image_str, latex_table_label_str = merge_horizontal_cell(
                    latex_table_str=latex_table_str,
                    rng=rng,
                    content=h_contents[rand_content_index],
                )
            else:
                raise ValueError("merge_method should choice from ['random', 'vertical', 'horizontal']")

            logger.debug(latex_table_image_str)

            Path(output_path).mkdir(exist_ok=True, parents=True)
            tables = run_table_detect(file_image)
            if tables:
                image = get_fit_size_latex_table_to_image(
                    latex_table_str=latex_table_image_str,
                    file_image=file_image,
                    table=tables[0],
                )
                _ = convert_latex_table_to_pandas(
                    latex_table_str=latex_table_label_str,
                    headers=True,
                )
                final_image = draw_table_bbox(src=file_image, tables=tables, margin=5)
                final_image = paste_image_with_table_bbox(src=final_image, dst=image, table=tables[0], margin=10)
                plt.imsave(Path(output_path, filename.stem + ".jpg"), final_image)
                with Path(output_path, filename.stem + ".txt").open("w", encoding="utf-8") as f:
                    f.write(latex_table_label_str)
            else:
                logger.info(f"Not detect table, so skip {filename.name}")
        except Exception as e:
            logger.exception(e)
            logger.error(f"{filename.name} have error")


if __name__ == "__main__":
    args = arg_parser()

    # Pre-process arguments
    check_arguments = [
        "seed",
        "h_contents",
        "v_contents",
        "specific_headers",
    ]
    for check_argument in check_arguments:
        if not getattr(args, check_argument):
            delattr(args, check_argument)

    if len(args.vertical) > 1:
        args.vertical = (
            [int(args.vertical[0]), int(args.vertical[1])]
            if int(args.vertical[0]) < int(args.vertical[1])
            else [int(args.vertical[1]), int(args.vertical[0])]
        )
    elif len(args.vertical) == 1:
        args.vertical = int(args.vertical[0])
    else:
        delattr(args, "vertical")

    args = vars(args)
    print(pprint.pformat(args))

    main(**args)
