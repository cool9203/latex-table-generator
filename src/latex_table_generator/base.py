# coding: utf-8

import io
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
from cv2.typing import MatLike
from img2table.tables.objects.extraction import ExtractedTable
from PIL import Image as PILImage

InputType = Union[str, Path, bytes, io.BytesIO, MatLike, PILImage.Image]


def crop_table_bbox(
    src: InputType,
    tables: List[ExtractedTable],
    margin: int,
    **kwds,
) -> List[MatLike]:
    img = get_image(src=src)
    crop_img = list()

    for table in tables:
        x1 = max(table.bbox.x1 - margin, 0)
        x2 = min(table.bbox.x2 + margin, img.shape[1])
        y1 = max(table.bbox.y1 - margin, 0)
        y2 = min(table.bbox.y2 + margin, img.shape[0])
        crop_img.append(img[y1:y2, x1:x2])

    return crop_img


def get_image(
    src: InputType,
) -> MatLike:
    # Instantiation of document, either an image or a PDF
    if isinstance(src, (MatLike, np.ndarray)):
        img = src
    elif isinstance(src, PILImage.Image):
        img = np.asarray(src)
    else:
        if isinstance(src, bytes):
            _src = src
        elif isinstance(src, io.BytesIO):
            src.seek(0)
            _src = src.read()
        elif isinstance(src, (str, Path)):
            with io.open(str(src), "rb") as f:
                _src = f.read()
        else:
            raise TypeError(f"Not implement image type: {type(src)}")
        img = cv2.imdecode(np.frombuffer(_src, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
