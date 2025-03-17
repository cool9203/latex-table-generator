# coding: utf-8

import argparse
import json
from pathlib import Path

import tqdm as TQDM
from PIL import Image


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert steel role to output label")
    parser.add_argument("-i", "--input_label_path", required=True, help="Input YOLO label data path")
    parser.add_argument("-o", "--output_path", required=True, help="Output path")

    args = parser.parse_args()
    return args


def cxcywh2xywh(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
    x1, y1 = x - w / 2, y - h / 2
    x2, y2 = x + w / 2, y + h / 2
    return x1, y1, x2, y2


def convert_yolo_label_to_labelstudio(
    input_label_path: str,
    output_path: str,
):
    label_studio_predictions = list()
    for file in Path(input_label_path).iterdir():
        if file.suffix.lower() not in [".txt"] or file.stem in ["classes"] or file.is_dir():
            continue

        with file.open(mode="r", encoding="utf-8") as f:
            yolo_labels = [line.split(" ") for line in f.readlines()]

        image = Image.open(Path(file.parent, f"{file.stem}.jpg"))
        label_studio_predictions.append(
            {
                "data": {
                    "image": f"/data/local-files/?d=tmpco-20250227/{file.stem}.jpg",
                },
                "predictions": [
                    {
                        "model_version": "one",
                        "score": 0.5,
                        "result": [
                            {
                                "id": f"result{i + 1}",
                                "type": "rectanglelabels",
                                "from_name": "label",
                                "to_name": "image",
                                "original_width": image.width,
                                "original_height": image.height,
                                "image_rotation": 0,
                                "value": {
                                    "rotation": 0,
                                    "x": (float(label[1]) - float(label[3]) / 2) * 100.0,
                                    "y": (float(label[2]) - float(label[4]) / 2) * 100.0,
                                    "width": float(label[3]) * 100.0,
                                    "height": float(label[4]) * 100.0,
                                    "rectanglelabels": ["text" if label[0] == "0" else "steel"],
                                },
                            }
                            for i, label in enumerate(yolo_labels)
                        ],
                    }
                ],
            }
        )

    with Path(output_path).open(mode="w", encoding="utf-8") as f:
        json.dump(
            obj=label_studio_predictions,
            fp=f,
        )


if __name__ == "__main__":
    args = arg_parse()
    convert_yolo_label_to_labelstudio(**vars(args))
