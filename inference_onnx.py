#!/usr/bin/env python
# -*- coding: utf-8 -*-
from models import DetectHandler
import torch


def main(args):
    import onnxruntime as rt
    import cv2
    import numpy as np
    from config import Config

    total_config = Config()
    dataset = args.dataset
    dataset_class = total_config.DATASETS[dataset]
    dataset_params = total_config.DATASET_PARAMS[dataset]
    dataset_instance = dataset_class(data_path=total_config.DATA_PATH)

    img = cv2.imread(args.image_path)
    assert img is not None

    ori_h, ori_w = img.shape[:2]

    h_ratio = ori_h / args.img_h
    w_ratio = ori_w / args.img_w

    processed_img = cv2.resize(img, (args.img_w, args.img_h))
    processed_img = processed_img / 255.0
    input_x = processed_img.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)

    sess = rt.InferenceSession(args.onnx_weight_file)

    assert len(sess.get_inputs()) == 1
    assert len(sess.get_outputs()) == 1

    input_name = sess.get_inputs()[0].name
    output_names = [elem.name for elem in sess.get_outputs()]
    predictions = sess.run(output_names, {input_name: input_x})[0]

    detect_handler = DetectHandler(
        num_classes=3, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, h_ratio=h_ratio, w_ratio=w_ratio,
    )
    bboxes, scores, classes = detect_handler(predictions)

    result = dataset_class.visualize_one_image_util(
        img, dataset_instance.classes, dataset_instance.colors, bboxes, classes,
    )
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_h", type=int, required=True, help="height size of the input")
    parser.add_argument("--img_w", type=int, required=True, help="width size of the input")
    parser.add_argument(
        "--dataset", type=str, required=True, help="name of the dataset to use",
    )
    parser.add_argument("--image_path", type=str, required=True, help="path to image")
    parser.add_argument("--onnx_weight_file", type=str, required=True)
    parser.add_argument("--conf_thresh", type=float, default=0.1)
    parser.add_argument("--nms_thresh", type=float, default=0.5)
    parsed_args = parser.parse_args()

    main(parsed_args)
