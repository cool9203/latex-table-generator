# coding: utf-8

import argparse

import gradio as gr
import pypandoc

from latex_table_generator import utils


def arg_parser() -> argparse.Namespace:
    """取得執行程式時傳遞的參數

    tutorial: https://docs.python.org/zh-tw/3/howto/argparse.html#
    reference: https://docs.python.org/zh-tw/3/library/argparse.html#nargs

    Returns:
        argparse.Namespace: 使用args.name取得傳遞的參數
    """

    parser = argparse.ArgumentParser(description="Pypandoc test web")

    parser.add_argument("--host", type=str, default=None, help="Server host")
    parser.add_argument("--port", type=int, default=None, help="Server port")

    args = parser.parse_args()

    return args


def convert_to_html(
    source: str,
    format: str,
    repair_latex: bool = False,
    full_border: bool = False,
    unsqueeze: bool = False,
) -> tuple[str, str]:
    response = ""
    if format in ["latex"]:
        if repair_latex:
            source = utils.convert_pandas_to_latex(
                df=utils.convert_latex_table_to_pandas(
                    latex_table_str=source,
                    headers=True,
                    unsqueeze=unsqueeze,
                ),
                full_border=full_border,
            )

    try:
        response = pypandoc.convert_text(
            source=source,
            to="html",
            format=format,
        )
    except Exception as e:
        pass

    return (
        response if "<table>" in response else f"格式有誤或是並非 {format}",
        source,
    )


def pypandoc_web(
    host: str = "0.0.0.0",
    port: int = 7860,
):
    with gr.Blocks(title="Pypandoc 渲染測試網站") as demo:
        gr.Markdown("## Pypandoc 渲染測試網站")

        with gr.Row():
            with gr.Column():
                source_text = gr.TextArea(label="Source")
                repair_source_text = gr.TextArea(label="Repair source")

            with gr.Column():
                html_table = gr.HTML(label="渲染結果")

        source_format = gr.Dropdown(
            [
                "latex",
                "markdown",
                "html",
            ],
            label="Source format",
        )
        repair_latex = gr.Checkbox(value=True, label="修復 latex")
        full_border = gr.Checkbox(label="修復 latex 表格全框線")
        unsqueeze = gr.Checkbox(label="修復 latex 並解開多行/列合併")
        submit_button = gr.Button("渲染")

        submit_button.click(
            convert_to_html,
            inputs=[
                source_text,
                source_format,
                repair_latex,
                full_border,
                unsqueeze,
            ],
            outputs=[html_table, repair_source_text],
        )

        demo.launch(
            server_name=host,
            server_port=port,
        )


if __name__ == "__main__":
    args = arg_parser()

    check_arguments = [
        "host",
        "port",
    ]

    # Pre-process arguments
    for check_argument in check_arguments:
        if hasattr(args, check_argument) and not getattr(args, check_argument):
            delattr(args, check_argument)

    pypandoc_web(**vars(args))
