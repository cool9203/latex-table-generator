# coding: utf-8

import subprocess
import traceback
from pathlib import Path

from matplotlib import pyplot as plt
from pdf2image import convert_from_path

from latex_table_generator.base import crop_table_bbox
from latex_table_generator.camelot_base import run_table_detect


def latex_table_to_image(
    latex_code,
    output_path="output.png",
    dpi=300,
    timeout: int = 10,
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

    # 寫入 LaTeX 文件
    tex_file.write_text(latex_code, encoding="utf-8")

    try:
        # 編譯 LaTeX 文件為 PDF
        try:
            subprocess.run(
                ["pdflatex", "-output-directory", str(temp_dir), str(tex_file)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
            print("Convert latex table to pdf success")
        except subprocess.CalledProcessError as e:
            print("Error in LaTeX compilation:", e.stderr.decode("utf-8"))
            return
        except subprocess.TimeoutExpired as e:
            print(e)
            return

        # 將 PDF 轉換為 PNG (使用 Pillow)
        try:
            images = convert_from_path(pdf_file, dpi=dpi, fmt="png")
            print("Convert pdf to image success")
            if isinstance(images, list) and images:
                image = images[0]
                tables = run_table_detect(image)
                crop_table_images = crop_table_bbox(src=image, tables=tables, margin=10)
                plt.imsave(str("output_path.png"), crop_table_images[0])
        except Exception as e:
            traceback.print_exception(e)
    except Exception:
        pass
    finally:
        # 清理臨時文件
        for file in temp_dir.iterdir():
            file.unlink()
        temp_dir.rmdir()


if __name__ == "__main__":
    # 使用範例
    # 定義 LaTeX 表達式
    latex_expression = r"""
    \documentclass{article}

    \usepackage{CJKutf8}

    \begin{document}
    \begin{CJK}{UTF8}{gbsn}

    \begin{tabular}{|c|c|c|c|c|c|c|c|c|c|}
    \hline 編號 & 組編號 & 號數 & 長A & 型狀/長度B & 長C & 總長度 & 支數 & 重量 & 備註 \\
    \hline 1 & 彎料 & \#5 & & 450 & & 480 & 10 & 75 & \\
    \hline 2 & 彎料 & \#4 & 80 & 80 & & 340 & 720 & 2433 & \\
    \hline 3 & 彎料 & \#4 & 65 & 80 & & 310 & 10 & 31 & \\
    \hline 4 & 彎料 & \#4 & 10 & 81 & 12 & 105 & 2800 & 2922 & \\
    \hline
    \end{tabular}

    \end{CJK}
    \end{document}"""

    latex_table_to_image(latex_expression, output_path="latex_formula.png")
