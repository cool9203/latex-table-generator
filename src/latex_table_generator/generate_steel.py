# coding: utf-8

import json
import logging
import random
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import List, Sequence, Tuple, Union

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

from latex_table_generator.base import image_resize

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel("INFO")


def _get_fit_font_size(
    text: str,
    size: Tuple[int, int, int, int],
    font: str,
) -> ImageFont:
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
    return _font


@dataclass
class SteelRole:
    x1: float
    x2: float
    y1: float
    y2: float
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


@dataclass
class Steel(ImageBase):
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

            ranges = [n for n in range(*role.range)]
            text += str(ranges[rng.randint(0, len(ranges) - 1)])
            if role.angle:
                text += "°"

            if role.after_choices:
                text += role.after_choices[rng.randint(0, len(role.after_choices) - 1)]

            _font = _get_fit_font_size(
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
        roles: List[SteelRole] = None,
        font: str = "mingliu.ttc",
        font_color: Union[str, Tuple[int, int, int]] = (0, 0, 0),
        size: Tuple[int, int] = None,
        rng: random.Random = None,
        max_fraction: float = 0.7,
        min_fraction: float = 0.3,
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

            ranges = [n for n in range(*role.range)]
            text += str(ranges[rng.randint(0, len(ranges) - 1)])
            if role.angle:
                text += "°"

            if role.after_choices:
                text += role.after_choices[rng.randint(0, len(role.after_choices) - 1)]

            start_x = role.x1 + rng.randint(int((role.x2 - role.x1) * min_fraction), int((role.x2 - role.x1) * max_fraction))
            start_y = role.y1 + rng.randint(int((role.y2 - role.y1) * min_fraction), int((role.y2 - role.y1) * max_fraction))
            _font = _get_fit_font_size(
                text=text,
                size=(start_x, start_y, role.x2, role.y2),
                font=font,
            )
            draw_image.text((start_x, start_y), text, fill=font_color, font=_font)
            label_texts.append(text)
        return (image_resize(src=image, size=size), " ".join(label_texts))


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
    from uuid import uuid4

    import tqdm as TQDM
    from imgaug import augmenters as iaa

    from latex_table_generator.base import get_image

    image_paths = "./steels"
    output_path = "./outputs/random-steels"
    image_size = (644, 644)
    count = 500
    rng = random.Random(42)

    image_augmenter = iaa.OneOf(
        [
            iaa.AdditiveGaussianNoise(scale=(0, 30)),
            iaa.AdditiveGaussianNoise(scale=(0, 30), per_channel=True),
            iaa.SaltAndPepper(p=(0, 0.1)),
            iaa.GaussianBlur(sigma=(0, 2.0)),
            iaa.JpegCompression(compression=(0, 85)),
            iaa.AverageBlur(k=(1, 5)),
        ],
    )

    Path(output_path).mkdir(parents=True, exist_ok=True)

    base_images = load_image(
        image_paths,
        rng=rng,
    )

    for steel in TQDM.tqdm(base_images):
        for i in TQDM.tqdm(range(count), leave=False):
            (generate_image, generate_image_label) = steel.random_generate(rng=rng)

            name = str(uuid4())
            save_path = Path(
                output_path,
                Path(steel.image_path).name.replace(Path(steel.image_path).suffix, ""),
            )
            generate_image_path = Path(save_path, f"{name}.png")
            generate_image_label_path = Path(save_path, f"{name}.txt")
            save_path.mkdir(exist_ok=True)

            generate_image = generate_image.convert("RGB")
            generate_image = image_resize(src=generate_image, size=image_size)
            generate_image = PILImage.fromarray(image_augmenter(images=[get_image(generate_image)])[0])
            generate_image.save(generate_image_path)

            with generate_image_label_path.open("w", encoding="utf-8") as f:
                f.write(generate_image_label)
