# coding: utf-8

import argparse
import pprint


def arg_parser() -> argparse.Namespace:
    """取得執行程式時傳遞的參數

    tutorial: https://docs.python.org/zh-tw/3/howto/argparse.html#
    reference: https://docs.python.org/zh-tw/3/library/argparse.html#nargs

    Returns:
        argparse.Namespace: 使用args.name取得傳遞的參數
    """

    parser = argparse.ArgumentParser(description="Auto generate latex table data")

    parser.add_argument("-i", "--input_path", type=str, nargs="+", required=True, help="Input path(folder)")
    parser.add_argument("-o", "--output_path", required=True, help="Output path")
    parser.add_argument(
        "-m",
        "--merge_method",
        type=str,
        choices=["random", "vertical", "horizontal", "hybrid"],
        default="random",
        help="Merge method",
    )
    parser.add_argument("-vc", "--v_contents", type=str, nargs="+", default=[], help="Merged cell content, will random choice")
    parser.add_argument("-hc", "--h_contents", type=str, nargs="+", default=[], help="Merged cell content, will random choice")
    parser.add_argument("-s", "--seed", type=str, default=None, help="Random seed")
    parser.add_argument("--specific_headers", type=str, nargs="+", default=[], help="Choice column name to merge")
    parser.add_argument("--vertical", type=str, nargs="+", default=[], help="Vertical length can be range")
    parser.add_argument("--tqdm", action="store_true", help="Use tqdm to show progress bar")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = arg_parser()

    # Pre-process arguments
    check_arguments = [
        "seed",
        "h_contents",
        "v_contents",
        "specific_headers",
    ]
    for check_argument in check_arguments:
        if not getattr(args, check_argument):
            delattr(args, check_argument)

    if len(args.vertical) > 1:
        args.vertical = (
            [int(args.vertical[0]), int(args.vertical[1])]
            if int(args.vertical[0]) < int(args.vertical[1])
            else [int(args.vertical[1]), int(args.vertical[0])]
        )
    elif len(args.vertical) == 1:
        args.vertical = int(args.vertical[0])
    else:
        delattr(args, "vertical")

    args = vars(args)
    print(pprint.pformat(args))

    from latex_table_generator.main import get_subfolder_path, main

    input_paths = args.pop("input_path")
    for input_path in input_paths:
        subfolder_paths = get_subfolder_path(input_path)
        print(f"subfolder_paths: {subfolder_paths}")
        for subfolder_path in subfolder_paths:
            main(
                input_path=subfolder_path,
                **args,
            )
