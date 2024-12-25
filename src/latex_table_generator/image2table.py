# coding: utf-8
# TODO: 隱性表格線 跟 顯性表格線 怎麼結合
# TODO: 要確認有些表格線框的歪歪的, 且好像有多框還怎麼樣的

import copy
import io
from typing import List

import cv2
from img2table.document import Image as ImageDoc
from img2table.tables.image import TableImage
from img2table.tables.objects.extraction import ExtractedTable
from matplotlib import pyplot as plt
from PIL import Image

from latex_table_generator.base import InputType, get_image


def run_table_detect(
    src: InputType,
    implicit_rows=False,
    implicit_columns=False,
    min_confidence=0,
    show_processed_image: bool = False,
    **kwds,
) -> List[ExtractedTable]:
    img = get_image(src=src)
    image_bytes = io.BytesIO()
    Image.fromarray(img).save(fp=image_bytes, format="png")
    document = ImageDoc(image_bytes)

    # image pre-process
    processed_img = img
    # gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

    # gray_img = cv2.erode(gray_img, kernel)  # 對黑色膨脹(侵蝕白色)
    # gray_img = cv2.dilate(gray_img, kernel)  # 對黑色侵蝕(白色膨脹)

    # processed_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)

    if show_processed_image:
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("origin")

        plt.subplot(1, 2, 2)
        plt.imshow(processed_img)
        plt.title("processed")

        plt.show()

    doc = TableImage(img=processed_img, **kwds)

    # Table extraction
    tables = doc.extract_tables(
        implicit_rows=implicit_rows,
        implicit_columns=implicit_columns,
        borderless_tables=False,
    )
    if not tables:
        print("borderless")
        tables = doc.extract_tables(
            implicit_rows=implicit_rows,
            implicit_columns=implicit_columns,
            borderless_tables=True,
        )

    _img = copy.deepcopy(processed_img)
    for line in doc.lines:
        if line.horizontal:
            cv2.rectangle(_img, (line.x1, line.y1), (line.x2, line.y2), (255, 0, 0), 2)
        elif line.vertical:
            cv2.rectangle(_img, (line.x1, line.y1), (line.x2, line.y2), (255, 0, 0), 2)

    if show_processed_image:
        plt.imshow(_img)
        plt.show()

    result = document.get_table_content(
        tables={0: tables},
        ocr=None,
        min_confidence=min_confidence,
    ).get(0)
    image_bytes.close()
    return result
