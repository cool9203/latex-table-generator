# coding: utf-8

import decimal
import json
import logging
import os
import random
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

from latex_table_generator import utils
from latex_table_generator.base import get_image, image_resize, paste_image, rotate_img_with_border

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

ctx = decimal.Context()
ctx.prec = 20


@dataclass
class Position:
    x1: float
    x2: float
    y1: float
    y2: float

    def to_list(self) -> Sequence[float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class SteelRole(Position):
    angle: bool = False
    range: Sequence[Union[int, float]] = field(default_factory=[1, 1000])
    choices: Sequence[str] = field(default_factory=[])
    before_choices: Sequence[str] = field(default_factory=[])
    after_choices: Sequence[str] = field(default_factory=[])


@dataclass
class ImageBase:
    origin_image: PILImage.Image
    image: PILImage.Image
    role_path: PathLike
    image_path: PathLike
    size: Tuple[int, int]


def _get_fit_font_size(
    text: str,
    size: Tuple[int, int, int, int],
    font: str,
) -> Tuple[ImageFont.FreeTypeFont, Position]:
    # Edit from: https://stackoverflow.com/a/61891053
    font_size = 1
    jump_size = 75
    _font = ImageFont.truetype(font, font_size)
    while True:
        text_bbox = _font.getbbox(text)
        logger.debug(f"text_bbox: {text_bbox}")
        logger.debug(f"size: {size}")
        logger.debug("-" * 25)

        if text_bbox[3] < (size[3] - size[1]) and text_bbox[2] < (size[2] - size[0]):
            font_size += jump_size
        else:
            jump_size = jump_size // 2
            font_size -= jump_size
        _font = ImageFont.truetype(font, font_size)

        if jump_size <= 1:
            break
    return (
        _font,
        Position(
            x1=text_bbox[0],
            x2=text_bbox[2],
            y1=text_bbox[1],
            y2=text_bbox[3],
        ),
    )


@dataclass
class Steel(ImageBase):
    position: Position = None
    rotate_angle: float = 0.0
    roles: List[SteelRole] = field(default_factory=[])
    rng: random.Random = None

    def __init__(
        self,
        image_path: PathLike,
        role_path: PathLike = None,
        rotate_angle: float = 0.0,
        size: Tuple[int, int] = None,
        rng: random.Random = None,
    ):
        self.image_path = Path(image_path)
        self.roles = list()
        if role_path:
            self.role_path = Path(role_path)
        elif Path(self.image_path.parent, f"{self.image_path.stem}.txt").exists():
            self.role_path = Path(self.image_path.parent, f"{self.image_path.stem}.txt")
        elif Path(self.image_path.parent, f"{self.image_path.stem}.json").exists():
            self.role_path = Path(self.image_path.parent, f"{self.image_path.stem}.json")
        with self.role_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("position", None):
            self.position = Position(**data.get("position"))

        for role in data.get("roles"):
            self.roles.append(SteelRole(**role))

        image = PILImage.open(image_path)
        self.origin_image = image
        self.image = self.hollow_image(image)
        self.size = size
        self.rotate_angle = rotate_angle
        self.rng = rng if rng else random.Random()

    def hollow_image(
        self,
        image: PILImage.Image,
    ) -> PILImage.Image:
        _image = image.copy()
        draw_image = ImageDraw.Draw(_image)
        for role in self.roles:
            draw_image.rectangle(((role.x1, role.y1), (role.x2, role.y2)), fill=(255, 255, 255))
        del draw_image
        return _image

    def draw_bbox(
        self,
        roles: List[SteelRole] = None,
    ) -> PILImage.Image:
        roles = roles if roles else self.roles
        image = self.origin_image.copy()
        draw_image = ImageDraw.Draw(image)
        for index, role in enumerate(roles):
            draw_image.text((role.x1, role.y1), str(index), fill=(0, 0, 0))
            draw_image.rectangle(((role.x1, role.y1), (role.x2, role.y2)), outline="red")
        del draw_image
        return image

    def generate(
        self,
        roles: List[SteelRole] = None,
        font: str = "mingliu.ttc",
        font_color: Union[str, Tuple[int, int, int]] = (0, 0, 0),
        size: Tuple[int, int] = None,
        rng: random.Random = None,
    ) -> Tuple[PILImage.Image, str]:
        rng = rng if rng else self.rng
        roles = roles if roles else self.roles
        size = size if size else self.size

        image = self.image.copy()
        draw_image = ImageDraw.Draw(image)
        label_texts = list()
        for role in roles:
            text = ""
            if role.before_choices:
                text += role.before_choices[rng.randint(0, len(role.before_choices) - 1)]

            ranges = [n for n in np.arange(*role.range)]
            text += str(ranges[rng.randint(0, len(ranges) - 1)])
            if role.angle:
                text += "°"

            if role.after_choices:
                text += role.after_choices[rng.randint(0, len(role.after_choices) - 1)]

            (_font, text_position) = _get_fit_font_size(
                text=text,
                size=(role.x1, role.y1, role.x2, role.y2),
                font=font,
            )
            draw_image.text((role.x1, role.y1), text, fill=font_color, font=_font)
            label_texts.append(text)
        del draw_image
        return (image_resize(src=image, size=size), " ".join(label_texts))

    def random_generate(
        self,
        count: int,
        roles: List[SteelRole] = None,
        font: str = "mingliu.ttc",
        font_color: Union[str, Tuple[int, int, int]] = (0, 0, 0),
        size: Tuple[int, int] = None,
        rng: random.Random = None,
        iterations: int = 100,
        steel_rotate: Sequence[int] = None,
        steel_flip: Optional[Literal["none", "random", "horizontal", "vertical", "all"]] = None,
        steel_size_scale: Sequence[int] = None,
        texts: List[str] = None,
    ) -> Tuple[PILImage.Image, List[Dict[str, Union[Position, str]]]]:
        steel_flip = steel_flip.lower() if steel_flip else steel_flip
        if steel_flip and steel_flip not in ["none", "random", "horizontal", "vertical", "all"]:
            raise ValueError(f"'steel_flip' not allow, should choice from {['none', 'random', 'horizontal', 'vertical', 'all']}")

        rng = rng if rng else self.rng
        roles = roles if roles else self.roles
        size = size if size else (self.size if self.size else self.image.size)  # (x, y)

        image = get_image(PILImage.new(mode="RGB", size=size, color=(255, 255, 255)))

        # Generate steel image
        steel_image = get_image(self.image.convert("RGB"))[
            self.position.y1 : self.position.y2, self.position.x1 : self.position.x2
        ]

        if steel_flip and steel_flip == "random":
            steel_flip = rng.choice(["horizontal", "vertical", "all"])

        if steel_flip and steel_flip == "vertical":
            steel_image = cv2.flip(src=steel_image, flipCode=0)
        elif steel_flip and steel_flip == "horizontal":
            steel_image = cv2.flip(src=steel_image, flipCode=1)
        elif steel_flip and steel_flip == "all":
            steel_image = cv2.flip(src=steel_image, flipCode=-1)

        if steel_rotate:
            angles = [n for n in np.arange(*list(steel_rotate))]
            angle = rng.choice(angles)
            steel_image = rotate_img_with_border(img=steel_image, angle=angle)
        if steel_size_scale:
            if rng.uniform(0, 4) >= 3:
                steel_image = cv2.resize(
                    src=steel_image,
                    dsize=(
                        min(int(steel_image.shape[1] * rng.uniform(steel_size_scale[0], steel_size_scale[1])), image.shape[1]),
                        min(int(steel_image.shape[0] * rng.uniform(steel_size_scale[0], steel_size_scale[1])), image.shape[0]),
                    ),  # (y, x)
                    interpolation=cv2.INTER_LANCZOS4,
                )
            else:
                scale = rng.uniform(steel_size_scale[0], steel_size_scale[1])
                steel_image = get_image(
                    image_resize(
                        src=steel_image,
                        size=(
                            int(min(steel_image.shape[0] * scale, image.shape[0])),
                            int(min(steel_image.shape[1] * scale, image.shape[1])),
                        ),  # (y, x)
                    )
                )

        # Paste steel image
        (image, steel_position) = paste_image(
            src=image,
            dst=steel_image,
            position=(
                rng.randint(0, image.shape[1] - steel_image.shape[1]) if image.shape[1] - steel_image.shape[1] > 0 else 0,
                rng.randint(0, image.shape[0] - steel_image.shape[0]) if image.shape[0] - steel_image.shape[0] > 0 else 0,
            ),
        )
        image = PILImage.fromarray(image)

        draw_image = ImageDraw.Draw(image)
        labels: List[Dict[str, Union[List[float], str]]] = [
            {
                "label": "steel",
                "position": steel_position,
            },
        ]
        for _ in range(count):
            text = format(ctx.create_decimal(repr(rng.uniform(1, 10000))), "f")
            precision = rng.randint(0, 3)
            angle = rng.uniform(0, 5)
            if angle >= 4:
                text = f"{text.split('.')[0]}°"
            else:
                text = text.split(".")[0] + ("." + text.split(".")[1][:precision] if precision else "")

            if texts and rng.uniform(0, 4) >= 3:
                if rng.randint(0, 1) == 1:
                    text += rng.choice(texts)
                else:
                    text = f"{rng.choice(texts)}{text}"

            for iteration in range(iterations):
                font_size = rng.randint(8, 72)
                x1 = rng.randint(0, image.width)
                y1 = rng.randint(0, image.height)
                _font = ImageFont.truetype(font, font_size)
                text_bbox = _font.getbbox(text)
                text_position = Position(
                    x1=x1,
                    x2=x1 + text_bbox[2],
                    y1=y1,
                    y2=y1 + text_bbox[3],
                )

                # Check out size
                if text_position.x2 > image.width or text_position.y2 > image.height:
                    continue

                # Check overlap
                is_overlap = False
                for label in labels:
                    if utils.check_overlap(a=text_position.to_list(), b=label.get("position")):
                        is_overlap = True

                if not is_overlap:
                    draw_image.text((x1, y1), text, fill=font_color, font=_font)
                    labels.append(
                        {
                            "label": text,
                            "position": text_position.to_list(),
                        }
                    )
                    break

        return (
            image_resize(src=image, size=(size[1], size[0])),
            labels,
        )


@dataclass
class NormalImage(ImageBase):
    label: str

    def __init__(
        self,
        image_path: PathLike,
        role_path: PathLike = None,
        size: Tuple[int, int] = None,
        **kwds,
    ):
        self.image_path = Path(image_path)
        self.roles = list()
        if role_path:
            self.role_path = Path(role_path)
        elif Path(self.image_path.parent, f"{self.image_path.stem}.txt").exists():
            self.role_path = Path(self.image_path.parent, f"{self.image_path.stem}.txt")
        with self.role_path.open("r", encoding="utf-8") as f:
            self.label = f.read()

        image = PILImage.open(image_path)
        self.origin_image = image
        self.size = size
        self.image = image_resize(image)

    def generate(self, **kwds) -> Tuple[PILImage.Image, str]:
        return (self.image, self.label)


def load_image(
    image_paths: List[PathLike] | str,
    extensions: List[str] = [".png", ".jpg", ".jpeg"],
    rng: random.Random = None,
) -> List[Steel]:
    rng = rng if rng else random.Random()
    image_paths = [image_paths] if isinstance(image_paths, str) else image_paths

    _image_paths = list()
    for image_path in image_paths:
        image_path = Path(image_path)
        if image_path.is_dir():
            for _extension in extensions:
                _image_paths += [p for p in Path(image_path).glob(f"*{_extension.lower()}")]
                _image_paths += [p for p in Path(image_path).glob(f"*{_extension.upper()}")]
        else:
            if image_path.suffix in extensions:
                _image_paths.append(image_path)

    logger.debug(f"load images: {_image_paths}")
    steels = list()
    for _image_path in _image_paths:
        try:
            steels.append(Steel(_image_path, rng=rng))
        except json.JSONDecodeError:
            steels.append(NormalImage(_image_path))
    return steels


if __name__ == "__main__":
    import argparse
    from uuid import uuid4

    import tqdm as TQDM
    from imgaug import augmenters as iaa

    parser = argparse.ArgumentParser(description="Generate steel image")
    parser.add_argument("-o", "--output_path", required=True, help="Output path")
    parser.add_argument("-i", "--input_paths", type=str, nargs="+", default=[], help="Input path(folder)")
    parser.add_argument("-c", "--count", type=int, default=None, help="Generate count")
    parser.add_argument("--image_augment", action="store_true", help="Use image augment")
    parser.add_argument("--steel_flip", type=str, choices=["random", "horizontal", "vertical", "all"], help="Steel flip")
    parser.add_argument("--steel_rotate", type=float, nargs="+", default=None, help="Rotate range")
    parser.add_argument("--steel_size_scale", type=float, nargs="+", default=None, help="Size scale range")
    parser.add_argument("--fonts_dir", type=str, default="./fonts", help="Fonts folder")

    args = parser.parse_args()

    _image_augmenter = iaa.OneOf(
        [
            iaa.AdditiveGaussianNoise(scale=(0, 30)),
            iaa.AdditiveGaussianNoise(scale=(0, 30), per_channel=True),
            iaa.SaltAndPepper(p=(0, 0.1)),
            iaa.GaussianBlur(sigma=(0, 2.0)),
            iaa.JpegCompression(compression=(0, 85)),
            iaa.AverageBlur(k=(1, 5)),
        ],
    )
    texts = [
        "準",
        "寬",
        "直料",
        "公分",
        "角度",
        "T頭",
        "公頭",
        "母頭",
        "cm",
        "\\",
        "/",
        "x",
        "X",
    ]

    rng = random.Random(os.getenv("SEED", None))
    fonts: list[str] = ["mingliu.ttc"]

    if Path(args.fonts_dir).exists():
        for p in Path(args.fonts_dir).iterdir():
            if p.suffix in [".ttf", ".otf", ".ttc"] and p.name not in fonts and p.resolve() not in fonts:
                fonts.append(p.resolve())

    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    base_images = load_image(
        args.input_paths,
        rng=rng,
    )

    for steel in TQDM.tqdm(base_images):
        for i in TQDM.tqdm(range(args.count), leave=False):
            number_count = int(Path(steel.image_path).name.replace("".join(Path(steel.image_path).suffixes), "").split("-")[1])
            number_count = rng.randint(max(1, number_count - 2), number_count + 2)
            params = {
                "rng": rng,
                "count": number_count,
                "iterations": 100,
                "font": fonts[rng.randint(0, len(fonts) - 1)],
                "texts": texts,
            }
            params.update(steel_flip=args.steel_flip) if args.steel_flip is not None else None
            params.update(steel_rotate=args.steel_rotate) if args.steel_rotate is not None else None
            params.update(steel_size_scale=args.steel_size_scale) if args.steel_size_scale is not None else None
            (generate_image, generate_image_label) = steel.random_generate(**params)

            name = str(uuid4())
            save_path = Path(
                args.output_path,
                Path(steel.image_path).name.replace(Path(steel.image_path).suffix, ""),
            )
            generate_image_path = Path(save_path, f"{name}.png")
            generate_image_label_path = Path(save_path, f"{name}.txt")
            save_path.mkdir(exist_ok=True)

            generate_image = generate_image.convert("RGB")
            generate_image = image_resize(src=generate_image, size=None)
            if args.image_augment:
                generate_image = PILImage.fromarray(_image_augmenter(images=[get_image(generate_image)])[0])
            generate_image.save(generate_image_path)

            with generate_image_label_path.open("w", encoding="utf-8") as f:
                json.dump(
                    obj=generate_image_label,
                    fp=f,
                    indent=4,
                    ensure_ascii=False,
                )
