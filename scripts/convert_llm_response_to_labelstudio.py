# coding: utf-8

import argparse
import json
from pathlib import Path

import opencc
import tqdm as TQDM

converter = opencc.OpenCC("s2t.json")

"""
[
    {
        "data": {
            "image": "/data/local-files/?d=tmpco-20250227/00sPvrfNQrjn_page3_0.jpg"
        },
        "predictions": [
            {
                "model_version": "one",
                "score": 1.0,
                "result": [
                    {
                        "value": "{\n\"買方公司統編\":\"\",\n\"賣方公司統編\":\"\",\n\"發票號碼\":\"\",\n\"發票日期\":\"\",\n\"發票金額\":\"\"\n}",
                        "id": "result1",
                        "from_name": "response",
                        "to_name": "image",
                        "type": "textarea",
                        "origin": "manual"
                    }
                ]
            }
        ]
    }
]
"""

_key2name = {
    "InvoiceNumber": "發票號碼",
    "CompanyName": "買受人名稱",
    "VnUniformNumber": "賣方公司統編",
    "BuUniformNumber": "買方公司統編",
    "InvoiceDate": "發票日期",
    "ItemName": "品項",
    "NetAmount": "未稅金額",
    "TaxAmount": "稅額",
    "TotalAmount": "總金額",
    "TotalAmountCH": "手寫金額",
}
_name2key = {v: k for k, v in _key2name.items()}
_default_predict = {
    "發票號碼": "",
    "買受人名稱": "",
    "賣方公司統編": "",
    "買方公司統編": "",
    "發票日期": "",
    "品項": [""],
    "未稅金額": "",
    "稅額": "",
    "總金額": "",
    "手寫金額": "",
}


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LLM response to label studio format")
    parser.add_argument("-i", "--input_path", required=True, help="Input llm response data path")
    parser.add_argument("-o", "--output_path", required=True, help="Output path")

    args = parser.parse_args()
    return args


def convert_llm_response_to_labelstudio(
    input_path: str,
    output_path: str,
):
    label_studio_predictions = list()
    with Path(input_path).open(mode="r", encoding="utf-8") as f:
        iter_data: list[dict[str, str | list[str] | dict[str, dict]]] = json.load(fp=f)

    for data in TQDM.tqdm(iter_data):
        llm_response = dict()
        try:
            for key, value in data["api_result"]["result"].items():
                if key not in _key2name:
                    raise ValueError(f"{key} not define")

                if isinstance(value, dict):
                    llm_response[_key2name[key]] = converter.convert(value["value"])
                elif isinstance(value, list):
                    llm_response[_key2name[key]] = [converter.convert(e["value"]) for e in value]
                else:
                    raise TypeError("LLM 輸出未知格式")
        except Exception as e:
            llm_response = dict()
            print(data)
            print(e)
            print("-" * 25)

        if not llm_response:
            llm_response = _default_predict

        label_studio_predictions.append(
            {
                "data": {
                    "image": f"/data/local-files/?d=newsoft-handwriting-20250317/{data['filename']}",
                },
                "predictions": (
                    [
                        {
                            "model_version": "one",
                            "score": 1.0,
                            "result": [
                                {
                                    "value": {"text": [json.dumps(llm_response, ensure_ascii=False, indent=4)]},
                                    "id": "result1",
                                    "from_name": "response",
                                    "to_name": "image",
                                    "type": "textarea",
                                    "origin": "manual",
                                }
                            ],
                        }
                    ]
                ),
            }
        )

    with Path(output_path).open(mode="w", encoding="utf-8") as f:
        json.dump(
            obj=label_studio_predictions,
            fp=f,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    args = arg_parse()
    convert_llm_response_to_labelstudio(**vars(args))
