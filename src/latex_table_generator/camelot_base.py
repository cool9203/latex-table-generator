# coding: utf-8

from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory

from camelot.parsers import Lattice
from img2table.tables.objects.extraction import BBox, ExtractedTable, TableCell
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
        parser.horizontal_text = []
        parser.vertical_text = []
        parser.rootname = "0"
        parser._generate_table_bbox()

        _tables = []
        # sort tables based on y-coord
        for table_idx, tk in enumerate(sorted(parser.table_bbox.keys(), key=lambda x: x[1], reverse=True)):
            cols, rows, v_s, h_s = parser._generate_columns_and_rows(table_idx, tk)
            table = parser._generate_table(table_idx, cols, rows, v_s=v_s, h_s=h_s)
            table._bbox = tk
            _tables.append(table)

        tables = list()
        for _table in _tables:
            tables.append(
                ExtractedTable(
                    bbox=BBox(
                        x1=int(_table._bbox[0]),
                        y1=int(parser.pdf_height - _table._bbox[3]),
                        x2=int(_table._bbox[2]),
                        y2=int(parser.pdf_height - _table._bbox[1]),
                    ),
                    title=None,
                    content=OrderedDict(
                        (
                            i,
                            [
                                TableCell(
                                    bbox=BBox(int(c[0]), parser.pdf_height - int(r[1]), int(c[1]), parser.pdf_height - int(r[0])),
                                    value=None,
                                )
                                for c in _table.cols
                            ],
                        )
                        for i, r in enumerate(_table.rows)
                    ),
                )
            )
        return tables
