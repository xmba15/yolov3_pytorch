#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys


_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
try:
    sys.path.append(os.path.join(_CURRENT_DIR, ".."))
    from models import YoloNet
except Exception as e:
    print(e)
    sys.exit(-1)


def main(args):
    import torch

    input_tensor = torch.randn(1, 3, args.img_h, args.img_w)

    params = {
        "anchors": [
            [0.016923076923076923, 0.027196652719665274],
            [0.018, 0.013855213023900243],
            [0.02355072463768116, 0.044977511244377814],
            [0.033722163308589605, 0.025525525525525526],
            [0.049479166666666664, 0.049575070821529746],
            [0.05290373906125696, 0.08290488431876607],
            [0.09375, 0.098],
            [0.150390625, 0.1838283227241353],
            [0.26125, 0.36540185240513895],
        ],
        "anchor_masks": [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        "num_classes": args.num_classes,
        "img_w": args.img_w,
        "img_h": args.img_h,
    }

    model = YoloNet(dataset_config=params)
    output = model(input_tensor)
    for idx, p in enumerate(output):
        print("branch_{}: {}".format(idx, p.size()))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_h", type=int, default=608)
    parser.add_argument("--img_w", type=int, default=608)
    parser.add_argument("--num_classes", type=int, default=80)
    parsed_args = parser.parse_args()

    main(parsed_args)
