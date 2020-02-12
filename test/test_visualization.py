#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2
import sys


_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
try:
    sys.path.append(os.path.join(_CURRENT_DIR, ".."))
    from config import Config
    from data_loader import UdacityDataset
except:
    print("cannot load module")
    exit(1)


def main():
    dt_config = Config()
    dataset = UdacityDataset(data_path=dt_config.DATA_PATH)
    img = dataset.visualize_one_image(10)
    cv2.imshow("visualized_bboxes", img)
    cv2.waitKey(0)
    cv2.imwrite("visualized_bboxes.png", img)


if __name__ == "__main__":
    main()
