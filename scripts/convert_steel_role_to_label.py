# coding: utf-8

import argparse
import json
import os
import shutil
from pathlib import Path
from uuid import uuid4


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert steel role to output label")
    parser.add_argument("-i", "--input_path", required=True, help="Input path")
    parser.add_argument("-o", "--output_path", required=True, help="Output path")

    args = parser.parse_args()
    return args


def convert_steel_role_to_label(
    input_path: os.PathLike,
    output_path: os.PathLike,
):
    iter_data = [p for p in Path(input_path).glob("*.txt")] + [p for p in Path(input_path).glob("*.json")]

    for role_file in iter_data:
        with role_file.open("r", encoding="utf-8") as f:
            roles = json.load(fp=f)

        labels = [
            {
                "label": "steel",
                "position": [roles["position"]["x1"], roles["position"]["y1"], roles["position"]["x2"], roles["position"]["y2"]],
            }
        ]

        labels += [
            {
                "label": "number",
                "position": [role["x1"], role["y1"], role["x2"], role["y2"]],
            }
            for role in roles["roles"]
        ]

        name = str(uuid4())
        save_path = Path(
            output_path,
            role_file.name.replace(role_file.suffix, ""),
        )
        image_path = Path(save_path, f"{name}.png")
        image_label_path = Path(save_path, f"{name}.txt")
        save_path.mkdir(parents=True, exist_ok=True)

        if Path(role_file.parent, f"{role_file.stem}.png").exists():
            shutil.copy(Path(role_file.parent, f"{role_file.stem}.png"), image_path)
        elif Path(role_file.parent, f"{role_file.stem}.PNG").exists():
            shutil.copy(Path(role_file.parent, f"{role_file.stem}.PNG"), image_path)

        with image_label_path.open("w", encoding="utf-8") as f:
            json.dump(
                obj=labels,
                fp=f,
                indent=4,
                ensure_ascii=False,
            )


if __name__ == "__main__":
    args = arg_parse()
    convert_steel_role_to_label(**vars(args))
