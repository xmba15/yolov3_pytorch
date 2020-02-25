#!/usr/bin/env python
# -*- coding: utf-8 -*-
from models import YoloNet
import torch.onnx


def main(args):
    import os
    from config import Config

    total_config = Config()
    if not args.dataset or args.dataset not in total_config.DATASETS.keys():
        raise Exception("specify one of the datasets to use in {}".format(list(total_config.DATASETS.keys())))
    if not args.snapshot or not os.path.isfile(args.snapshot):
        raise Exception("invalid snapshot")

    dataset = args.dataset
    dataset_class = total_config.DATASETS[dataset]
    dataset_params = total_config.DATASET_PARAMS[dataset]
    model = YoloNet(dataset_config=dataset_params)
    model.load_state_dict(torch.load(args.snapshot)["state_dict"])
    model.eval()

    batch_size = 1
    x = torch.randn(batch_size, 3, dataset_params["img_h"], dataset_params["img_w"])
    torch.onnx.export(
        model,
        x,
        args.onnx_weight_file,
        verbose=True,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        opset_version=11,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, type=str)
    parser.add_argument(
        "--dataset", type=str, required=True, help="name of the dataset to use",
    )
    parser.add_argument("--onnx_weight_file", type=str, default="output.onnx")
    parsed_args = parser.parse_args()

    main(parsed_args)
