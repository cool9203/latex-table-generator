# coding: utf-8

import argparse
import logging
import os
import random
from os import PathLike
from pathlib import Path
from typing import List

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
    parser.add_argument("-c", "--contents", type=str, nargs="+", default=[], help="Merged cell content, will random choice")
    parser.add_argument("-s", "--seed", type=str, default=None, help="Random seed")

    args = parser.parse_args()

    return args


def main(
    input_path: PathLike,
    output_path: PathLike,
    merge_method: str = "random",
    contents: List[str] = ["彎鉤", "鋼材筋"],
    **kwds,
):
    from matplotlib import pyplot as plt

    from latex_table_generator.base import crop_table_bbox, draw_table_bbox, paste_image_with_table_bbox
    from latex_table_generator.main import (
        PILImage,
        latex_table_to_image,
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
        rand_content_index = rng.randint(0, len(contents) - 1)

        with filename.open("r", encoding="utf-8") as f:
            latex_table_str = f.read()

        try:
            if merge_method == "random":
                _rand_num = rng.randint(0, 1)
                if _rand_num == 0:
                    latex_table_image_str, latex_table_label_str = merge_horizontal_cell(
                        latex_table_str,
                        rng=rng,
                        content=contents[rand_content_index],
                    )
                else:
                    latex_table_image_str, latex_table_label_str = merge_vertical_cell(
                        latex_table_str,
                        rng=rng,
                        content=contents[rand_content_index],
                    )
            elif merge_method == "vertical":
                latex_table_image_str, latex_table_label_str = merge_vertical_cell(
                    latex_table_str,
                    rng=rng,
                    content=contents[rand_content_index],
                )
            elif merge_method == "horizontal":
                latex_table_image_str, latex_table_label_str = merge_horizontal_cell(
                    latex_table_str,
                    rng=rng,
                    content=contents[rand_content_index],
                )
            else:
                raise ValueError("merge_method should choice from ['random', 'vertical', 'horizontal']")

            logger.debug(latex_table_image_str)
            logger.debug(latex_table_label_str)

            Path(output_path).mkdir(exist_ok=True, parents=True)

            image = latex_table_to_image(
                latex_table_image_str,
            )
            label = latex_table_to_image(
                latex_table_label_str,
            )

            if image and label:
                tables = run_table_detect(file_image)
                image_tables = run_table_detect(image)
                crop_table_images = crop_table_bbox(src=image, tables=image_tables, margin=10)
                if tables and crop_table_images:
                    final_image = draw_table_bbox(src=file_image, tables=tables, margin=5)
                    final_image = paste_image_with_table_bbox(
                        src=final_image, dst=crop_table_images[0], table=tables[0], margin=10
                    )
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
    if not args.seed:
        delattr(args, "seed")
    if not args.contents:
        delattr(args, "contents")

    main(**vars(args))
