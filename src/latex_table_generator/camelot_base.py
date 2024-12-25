# coding: utf-8

from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory

from camelot.parsers import Lattice
from img2table.tables.objects.extraction import BBox, ExtractedTable
from PIL import Image

from latex_table_generator.base import (
    InputType,
    get_image,
)


def run_table_detect(
    src: InputType,
    **kwds,
):
    img = get_image(src=src)
    with TemporaryDirectory() as tempdir:
        _path = Path(tempdir, "temp.png")
        img = Image.fromarray(img)
        img.save(fp=_path, format="png")

        parser = Lattice(**kwds)
        parser.pdf_width = img.width
        parser.pdf_height = img.height
        parser.imagename = _path
        parser._generate_table_bbox()

        tables = list()
        for points, joints in parser.table_bbox.items():
            tables.append(
                ExtractedTable(
                    bbox=BBox(
                        x1=int(points[0]),
                        y1=int(parser.pdf_height - points[3]),
                        x2=int(points[2]),
                        y2=int(parser.pdf_height - points[1]),
                    ),
                    title=None,
                    content=OrderedDict(),
                )
            )
        return tables
