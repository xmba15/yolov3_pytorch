#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2
import random
import numpy as np
import json
from .dataset_base import DatasetBase, DatasetConfigBase


class SwitchDatasetConfig(DatasetConfigBase):
    def __init__(self):
        super(SwitchDatasetConfig, self).__init__()

        self.CLASSES = [
            "switch-unknown",
            "switch-right",
            "switch-left",
        ]

        self.COLORS = DatasetConfigBase.generate_color_chart(self.num_classes)


_switch_config = SwitchDatasetConfig()


class SwitchDataset(DatasetBase):
    __name__ = "switch_dataset"

    def __init__(
        self,
        data_path,
        classes=_switch_config.CLASSES,
        colors=_switch_config.COLORS,
        phase="train",
        transform=None,
        shuffle=True,
        random_seed=2000,
        normalize_bbox=False,
        bbox_transformer=None,
        multiscale=False,
        resize_after_batch_num=10,
        train_val_ratio=0.9,
    ):
        super(SwitchDataset, self).__init__(
            data_path,
            classes=classes,
            colors=colors,
            phase=phase,
            transform=transform,
            shuffle=shuffle,
            normalize_bbox=normalize_bbox,
            bbox_transformer=bbox_transformer,
            multiscale=multiscale,
            resize_after_batch_num=resize_after_batch_num,
        )

        assert os.path.isdir(data_path)
        assert phase in ("train", "val")

        self._data_path = os.path.join(data_path, "switch_detection/data")
        assert os.path.isdir(self._data_path)

        self._phase = phase
        self._transform = transform

        _imgs_path = os.path.join(self._data_path, "imgs")
        _jsons_path = os.path.join(self._data_path, "labels")

        _all_image_paths = DatasetBase.get_all_files_with_format_from_path(_imgs_path, ".jpg")
        _all_json_paths = DatasetBase.get_all_files_with_format_from_path(_jsons_path, ".json")

        assert len(_all_image_paths) == len(_all_json_paths)

        _image_paths = [os.path.join(_imgs_path, elem) for elem in _all_image_paths]
        _targets = [self._load_one_json(os.path.join(_jsons_path, elem)) for elem in _all_json_paths]

        zipped = list(zip(_image_paths, _targets))
        random.seed(random_seed)
        random.shuffle(zipped)
        _image_paths, _targets = zip(*zipped)

        _train_len = int(train_val_ratio * len(_image_paths))
        if self._phase == "train":
            self._image_paths = _image_paths[:_train_len]
            self._targets = _targets[:_train_len]
        else:
            self._image_paths = _image_paths[_train_len:]
            self._targets = _targets[_train_len:]

    def _load_one_json(self, json_path):
        bboxes = []
        labels = []

        p_json = json.load(open(json_path, "r"))

        for obj in p_json["objects"]:
            if "boundingbox" in obj:
                x_min, y_min, x_max, y_max = obj["boundingbox"]
                label_text = obj["label"]
                label_idx = self._classes.index(label_text)

                bboxes.append([x_min, y_min, x_max, y_max])
                labels.append(label_idx)

        return [bboxes, labels]
