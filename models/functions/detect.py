#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from .utils import nms
from .new_types import BboxType


__all__ = ["DetectHandler"]


class DetectHandler(Function):
    def __init__(
        self, num_classes: int, conf_thresh: float, nms_thresh: float, h_ratio: float = None, w_ratio: float = None
    ):
        super(DetectHandler, self).__init__()
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        self.h_ratio = h_ratio
        self.w_ratio = w_ratio

    def __call__(self, predictions: torch.Tensor):
        if isinstance(predictions, np.ndarray):
            predictions = torch.FloatTensor(predictions)

        bboxes = predictions[..., :4].squeeze_(dim=0)
        scores = predictions[..., 4].squeeze_(dim=0)
        classes_one_hot = predictions[..., 5:].squeeze_(dim=0)
        classes = torch.argmax(classes_one_hot, dim=1)

        bboxes, scores, classes = nms(
            bboxes,
            scores,
            classes,
            num_classes=self.num_classes,
            bbox_mode=BboxType.CXCYWH,
            conf_thresh=self.conf_thresh,
            nms_thresh=self.nms_thresh,
        )

        if self.h_ratio is not None and self.w_ratio is not None:
            bboxes[..., [0, 2]] *= self.w_ratio
            bboxes[..., [1, 3]] *= self.h_ratio

        return bboxes, scores, classes
