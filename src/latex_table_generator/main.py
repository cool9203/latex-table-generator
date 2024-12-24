# coding: utf-8

import subprocess
from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image


def latex_to_image(
    latex_code,
    output_path="output.png",
    dpi=300,
    timeout: int = 3,
):
    """
    將 LaTeX 公式轉換為 PNG 圖片
    :param latex_code: LaTeX 表達式 (例如：r"$E=mc^2$")
    :param output_path: 輸出的圖片路徑 (預設為 output.png)
    :param dpi: 圖片解析度 (預設為 300 DPI)
    """
    # 創建臨時目錄
    temp_dir = Path("temp_latex")
    temp_dir.mkdir(exist_ok=True)

    # 定義文件路徑
    tex_file = temp_dir / "formula.tex"
    pdf_file = temp_dir / "formula.pdf"
    png_file = temp_dir / "formula.png"

    # 寫入 LaTeX 文件
    tex_file.write_text(latex_code, encoding="utf-8")

    # 編譯 LaTeX 文件為 PDF
    try:
        subprocess.run(
            ["pdflatex", "-output-directory", str(temp_dir), str(tex_file)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as e:
        print("Error in LaTeX compilation:", e.stderr.decode("utf-8"))
        return
    except subprocess.TimeoutExpired as e:
        print(e)
        return

    # 將 PDF 轉換為 PNG (使用 Pillow)
    try:
        images = convert_from_path(pdf_file, dpi=dpi)
        if isinstance(images, list) and images:
            images[0].save(output_path)
            print(f"LaTeX 表達式已成功轉換為圖片: {output_path}")
    except Exception as e:
        print("Error in converting PDF to PNG:", e)

    # 清理臨時文件
    for file in temp_dir.iterdir():
        file.unlink()
    temp_dir.rmdir()


# 使用範例
# 定義 LaTeX 表達式
latex_expression = r"""
\documentclass{article}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}

\begin{document}
\begin{tabular}{|c|c|c|}
\hline 0 & 1 & 2 \\
\hline & & \\
\hline
\end{tabular}
\end{document}"""

latex_to_image(latex_expression, output_path="latex_formula.png")
