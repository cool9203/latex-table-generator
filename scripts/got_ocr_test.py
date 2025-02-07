import argparse
import base64
import io
import json
import os
import pprint
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import gradio as gr
import httpx
import numpy as np
import opencc
import pandas as pd
import pypandoc
import spaces
import torch
from flask import Flask, jsonify, request
from PIL import Image
from swift.llm import InferRequest, ModelType, PtEngine, RequestConfig, TemplateType, get_model_tokenizer, get_template
from swift.tuners import Swift
from swift.utils import seed_everything
from transformers import AutoModel, AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
system = "        You should follow the instructions carefully and explain your answers in detail."
history = []
query = "<image>OCR with format:"

# Define upload and results folders
UPLOAD_FOLDER = "./uploads"
RESULTS_FOLDER = "./results"
app = Flask(__name__)
app.static_folder = "/home/iii/static"

converter = opencc.OpenCC("s2t")


(model, tokenizer, engine, template) = (None, None, None, None)


def arg_parser() -> argparse.Namespace:
    """取得執行程式時傳遞的參數

    tutorial: https://docs.python.org/zh-tw/3/howto/argparse.html#
    reference: https://docs.python.org/zh-tw/3/library/argparse.html#nargs

    Returns:
        argparse.Namespace: 使用args.name取得傳遞的參數
    """

    parser = argparse.ArgumentParser(description="Convert latex table to markdown")

    parser.add_argument("-m", "--model_name_or_path", required=True, help="Model name or path")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=30308, help="Server port")

    args = parser.parse_args()

    return args


# Convert image to base64
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


@spaces.GPU
def run_GOT(
    image,
    got_mode,
    crop_table_status,
    crop_table_padding,
    fine_grained_mode="",
    ocr_color="",
    ocr_box="",
):
    unique_id = str(uuid.uuid4())
    image_path = Path(UPLOAD_FOLDER, f"{unique_id}.png")
    crop_image_path = Path(UPLOAD_FOLDER, f"{unique_id}-crop.png")
    crop_image = image
    origin_response = ""
    html_response = ""

    shutil.copy(image, str(image_path))

    try:
        if crop_table_status:
            resp = httpx.post(
                "http://10.70.0.232:9999/upload",
                files={"file": image_path.open("rb")},
                data={
                    "action": "crop",
                    "padding": crop_table_padding,
                },
            )

            for _, crop_image_base64 in resp.json().items():
                crop_image_data = base64.b64decode(crop_image_base64)
                crop_image = Image.open(io.BytesIO(crop_image_data))

                with image_path.open("wb") as f:
                    f.write(crop_image_data)
                break

        if got_mode == "OCR":
            res = model.chat(tokenizer, str(image_path), ocr_type="ocr")
            return converter.convert(res), None, crop_image
        elif got_mode == "OCR II":
            infer_request = InferRequest(
                messages=[
                    {
                        "role": "system",
                        "content": "        You should follow the instructions carefully and explain your answers in detail.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": str(crop_image_path) if crop_image_path.exists() else str(image_path),
                            },
                            {"type": "text", "text": "OCR with format:"},
                        ],
                    },
                ]
            )
            request_config = RequestConfig(max_tokens=1280, temperature=0)
            resp_list = engine.infer([infer_request], request_config)
            response = resp_list[0].choices[0].message.content

            # Process the LaTeX code response
            origin_response = response.replace("罩位重", "單位重").replace(
                "彐鴉垂直向", "彎鉤垂直向"
            )  # Add all replacements as necessary

            # Generate timestamp for filenames
            timestamp = str(int(time.time()))

            # Paths for the HTML and CSV files in the static folder
            html_file_path = Path(app.static_folder, f"{timestamp}.html")

            # Convert LaTeX to HTML
            try:
                html_table = pypandoc.convert_text(origin_response, "html", format="latex")
                html_content = f"""
            <!DOCTYPE html>
            <html lang="zh-TW">
            <head>
                <meta charset="UTF-8">
                <title>資策會 - 佩波表格影像轉文字 - 結果展示</title>
                <style>
                    table, th, td {{ border: 1px solid black; border-collapse: collapse; padding: 8px; }}
                </style>
            </head>
            <body>
            <div>{html_table}</div><br>
            </body>
            </html>"""
            except Exception as e:
                print("Error converting LaTeX to HTML:", e)
                raise e

            # Save the HTML content to a file
            with html_file_path.open("w", encoding="utf-8") as html_file:
                html_file.write(html_content)

            encoded_html = converter.convert(base64.b64encode(converter.convert(html_content).encode("utf-8")).decode("utf-8"))
            download_link = f'<a href="data:text/html;base64,{encoded_html}" download="result_{unique_id}.html">下載結果</a>'
            html_response = f"{download_link}<br>{html_content}"
    except Exception as e:
        html_response = "推論輸出非 latex"
    finally:
        image_path.unlink(missing_ok=True)
        crop_image_path.unlink(missing_ok=True)
    return (origin_response, html_response, crop_image)


# Update UI elements based on task selection
def task_update(task):
    if "fine-grained" in task:
        return [
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        ]
    else:
        return [
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        ]


def fine_grained_update(task):
    if task == "box":
        return [
            gr.update(visible=False, value=""),
            gr.update(visible=True),
        ]
    elif task == "color":
        return [
            gr.update(visible=True),
            gr.update(visible=False, value=""),
        ]


# Cleanup old files
def cleanup_old_files():
    current_time = time.time()
    for folder in [UPLOAD_FOLDER, RESULTS_FOLDER]:
        for file_path in Path(folder).glob("*"):
            if current_time - file_path.stat().st_mtime > 3600:  # 1 hour
                file_path.unlink()


def main(
    model_name_or_path: str,
    host: str,
    port: int,
):
    global model, tokenizer, template, engine
    model, tokenizer = get_model_tokenizer(
        "stepfun-ai/GOT-OCR2_0", None, model_kwargs={"device_map": "cuda:0"}, revision="master"
    )
    model = Swift.from_pretrained(model, model_name_or_path, inference_mode=True)
    model.to("cuda")
    model.requires_grad_(False)

    model.generation_config.max_new_tokens = 2048
    model.generation_config.bos_token_id = 151643
    model.generation_config.eos_token_id = 151645
    model.generation_config.pad_token_id = 151643
    template = get_template(TemplateType.got_ocr2, tokenizer)
    seed_everything(42)
    engine = PtEngine.from_model_template(model, template)

    # Create folders if they don't exist
    for folder in [UPLOAD_FOLDER, RESULTS_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    with gr.Blocks(title="GOT-OCR 生成表格測試網站") as demo:
        gr.Markdown("## GOT-OCR 生成表格測試網站")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="filepath", label="上傳圖片")

            with gr.Column():
                task_dropdown = gr.Dropdown(
                    choices=[
                        "OCR",
                        "OCR II",
                    ],
                    label="OCR模型",
                    value="OCR II",
                )
                crop_table_status = gr.Checkbox(label="是否自動偵測表格", value=True)
                crop_table_padding = gr.Slider(label="偵測表格裁切框 padding", value=-60, minimum=-300, maximum=300, step=1)

        submit_button = gr.Button("生成表格")
        ocr_result = gr.Textbox(label="生成的文字輸出")

        with gr.Row():
            with gr.Column():
                crop_table_result = gr.Image(label="偵測表格結果")

            with gr.Column():
                html_result = gr.HTML(label="生成的表格輸出", show_label=True)

        task_dropdown.change(task_update, inputs=[task_dropdown], outputs=[])

        submit_button.click(
            run_GOT,
            inputs=[image_input, task_dropdown, crop_table_status, crop_table_padding],
            outputs=[ocr_result, html_result, crop_table_result],
        )
    demo.launch(server_name=host, server_port=port)


if __name__ == "__main__":
    args = arg_parser()
    arg_dict = vars(args)
    print(pprint.pformat(arg_dict))

    cleanup_old_files()

    main(**arg_dict)
