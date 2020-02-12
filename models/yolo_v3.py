#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from .yolo_layer import YoloLayer
from .layers import Darknet, PreDetectionConvGroup, UpsampleGroup


__all__ = ["YoloNet"]

_YOLOLAYER_PARAMS = {
    "lambda_xy": 1,
    "lambda_wh": 1,
    "lambda_conf": 1,
    "lambda_cls": 1,
    "obj_scale": 1,
    "noobj_scale": 1,
    "ignore_thres": 0.7,
}


class YoloNet(nn.Module):
    def __init__(self, dataset_config, yololayer_params=_YOLOLAYER_PARAMS):
        super(YoloNet, self).__init__()

        self._yololayer_params = yololayer_params
        self._all_anchors = [
            [int(w_e * dataset_config["img_w"]), int(h_e * dataset_config["img_h"])]
            for (w_e, h_e) in dataset_config["anchors"]
        ]
        self._anchors_masks = dataset_config["anchor_masks"]
        assert len(self._anchors_masks) > 0
        self._num_anchors_each_layer = len(self._anchors_masks[0])

        self._num_classes = dataset_config["num_classes"]

        self.stat_keys = [
            "loss",
            "loss_x",
            "loss_y",
            "loss_w",
            "loss_h",
            "loss_conf",
            "loss_cls",
            "nCorrect",
            "nGT",
            "recall",
        ]

        self.feature = Darknet([1, 2, 8, 8, 4])
        self.feature.add_cached_out(61)
        self.feature.add_cached_out(36)

        self.pre_det1 = PreDetectionConvGroup(
            1024, 512, num_classes=self._num_classes, num_anchors=self._num_anchors_each_layer
        )
        self.pre_det1.add_cached_out(-3)

        self.up1 = UpsampleGroup(512)
        self.pre_det2 = PreDetectionConvGroup(
            768, 256, num_classes=self._num_classes, num_anchors=self._num_anchors_each_layer
        )
        self.pre_det2.add_cached_out(-3)

        self.up2 = UpsampleGroup(256)
        self.pre_det3 = PreDetectionConvGroup(
            384, 128, num_classes=self._num_classes, num_anchors=self._num_anchors_each_layer
        )

        self.yolo_layers = [
            YoloLayer(
                anchors_all=self._all_anchors,
                anchors_mask=anchors_mask,
                num_classes=self._num_classes,
                lambda_xy=self._yololayer_params["lambda_xy"],
                lambda_wh=self._yololayer_params["lambda_wh"],
                lambda_conf=self._yololayer_params["lambda_conf"],
                lambda_cls=self._yololayer_params["lambda_cls"],
                obj_scale=self._yololayer_params["obj_scale"],
                noobj_scale=self._yololayer_params["noobj_scale"],
                ignore_thres=self._yololayer_params["ignore_thres"],
            )
            for anchors_mask in self._anchors_masks
        ]

    def forward(self, x: torch.Tensor, target=None):
        img_dim = (x.shape[3], x.shape[2])
        out = self.feature(x)
        dets = []

        # Detection layer 1
        out = self.pre_det1(out)
        dets.append(self.yolo_layers[0](out, img_dim, target))

        # Upsample 1
        r_head1 = self.pre_det1.get_cached_out(-3)
        r_tail1 = self.feature.get_cached_out(61)
        out = self.up1(r_head1, r_tail1)

        # Detection layer 2
        out = self.pre_det2(out)
        dets.append(self.yolo_layers[1](out, img_dim, target))

        # Upsample 2
        r_head2 = self.pre_det2.get_cached_out(-3)
        r_tail2 = self.feature.get_cached_out(36)
        out = self.up2(r_head2, r_tail2)

        # Detection layer 3
        out = self.pre_det3(out)
        dets.append(self.yolo_layers[2](out, img_dim, target))

        if target is not None:
            loss, *out = [sum(det) for det in zip(dets[0], dets[1], dets[2])]

            self.stats = dict(zip(self.stat_keys, out))
            self.stats["recall"] = self.stats["nCorrect"] / self.stats["nGT"] if self.stats["nGT"] else 0
            return loss
        else:
            return torch.cat(dets, 1)
