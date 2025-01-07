# coding: utf-8

import io
import random
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
from cv2.typing import MatLike
from img2table.document.base.rotation import estimate_skew, get_connected_components, get_relevant_angles
from img2table.tables.objects.extraction import BBox, ExtractedTable
from PIL import Image as PILImage

InputType = Union[str, Path, bytes, io.BytesIO, MatLike, PILImage.Image]


def run_random_crop_rectangle(
    src: InputType,
    min_crop_size: Union[float, int],
    rng: random.Random = None,
) -> List[ExtractedTable]:
    # Pre-process min_crop_size
    if isinstance(min_crop_size, float):
        if min_crop_size < 1.0 and min_crop_size > 0.0:
            min_crop_size = src.shape[0] * src.shape[1] * min_crop_size
        min_crop_size = int(min_crop_size)

    while True:
        x1 = rng.randint(0, src.shape[1] // 6)
        y1 = rng.randint(0, src.shape[0] // 6)
        x2 = rng.randint(src.shape[1] // 6, src.shape[1])
        y2 = rng.randint(src.shape[0] // 6, src.shape[0])

        if min_crop_size is None or (x2 - x1) * (y2 - y1) >= min_crop_size:
            break
    return [
        ExtractedTable(
            bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
            title=None,
            content=[],
        )
    ]


def rotate_img_with_border(
    img: np.ndarray,
    angle: float,
    background_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    Rotate an image of the defined angle and add background on border
    :param img: image array
    :param angle: rotation angle
    :param background_color: background color for borders after rotation
    :return: rotated image array
    """
    # Compute image center
    height, width = (img.shape[0], img.shape[1])
    image_center = (width // 2, height // 2)

    # Compute rotation matrix
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    # Get rotated image dimension
    bound_w = int(height * abs(rotation_mat[0, 1]) + width * abs(rotation_mat[0, 0]))
    bound_h = int(height * abs(rotation_mat[0, 0]) + width * abs(rotation_mat[0, 1]))

    # Update rotation matrix
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # Create rotated image with white background
    rotated_img = cv2.warpAffine(
        img, rotation_mat, (bound_w, bound_h), borderMode=cv2.BORDER_CONSTANT, borderValue=background_color
    )
    return rotated_img


def fix_rotation_image(
    img: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    """
    Fix rotation of input image (based on https://www.mdpi.com/2079-9292/9/1/55) by at most 45 degrees
    :param img: image array
    :return: rotated image array and boolean indicating if the image has been rotated
    """
    # Get connected components of the images
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cc_centroids, ref_height, thresh = get_connected_components(img=gray)

    # Check number of centroids
    if len(cc_centroids) < 2:
        return img, False

    # Compute most likely angles from connected components
    angles = get_relevant_angles(centroids=cc_centroids, ref_height=ref_height)
    # Estimate skew
    skew_angle = estimate_skew(angles=angles, thresh=thresh)

    if abs(skew_angle) > 0:
        return rotate_img_with_border(img=img, angle=skew_angle), True

    return img, False


def paste_image_with_table_bbox(
    src: InputType,
    dst: InputType,
    table: ExtractedTable,
    **kwds,
) -> MatLike:
    img = get_image(src=src)

    x1 = max(table.bbox.x1, 0)
    x2 = x1 + dst.shape[1]
    y1 = max(table.bbox.y1, 0)
    y2 = y1 + dst.shape[0]
    img[y1:y2, x1:x2] = dst

    return img


def draw_table_bbox(
    src: InputType,
    tables: List[ExtractedTable],
    margin: int,
    color: Tuple[int, int, int] = (255, 255, 255),
    **kwds,
) -> MatLike:
    img = get_image(src=src)

    for table in tables:
        x1 = max(table.bbox.x1 - margin, 0)
        x2 = min(table.bbox.x2 + margin, img.shape[1])
        y1 = max(table.bbox.y1 - margin, 0)
        y2 = min(table.bbox.y2 + margin, img.shape[0])
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

    return img


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
        img = np.asarray(src).astype(np.uint8)  # Reference: https://github.com/opencv/opencv/issues/24522#issuecomment-1972141659
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
