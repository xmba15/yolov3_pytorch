#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import json
from .dataset_base import DatasetBase, DatasetConfigBase


class BirdDatasetConfig(DatasetConfigBase):
    def __init__(self):
        super(BirdDatasetConfig, self).__init__()

        self.CLASSES = [
            "bird",
        ]

        self.COLORS = DatasetConfigBase.generate_color_chart(self.num_classes)


_bird_config = BirdDatasetConfig()


class BirdDataset(DatasetBase):
    __name__ = "bird_dataset"

    def __init__(
        self,
        data_path,
        classes=_bird_config.CLASSES,
        colors=_bird_config.COLORS,
        phase="train",
        transform=None,
        shuffle=True,
        random_seed=2000,
        normalize_bbox=False,
        bbox_transformer=None,
        multiscale=False,
        resize_after_batch_num=10,
    ):
        super(BirdDataset, self).__init__(
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
        assert phase in ("train", "val", "test")

        self._data_path = os.path.join(data_path, "bird_dataset")
        assert os.path.isdir(self._data_path)

        self._phase = phase
        self._transform = transform

        if self._phase == "test":
            self._image_path_base = os.path.join(self._data_path, "test")
            self._image_paths = sorted(
                [os.path.join(self._image_path_base, image_path) for image_path in os.listdir(self._image_path_base)]
            )
        else:
            self._train_path_base = os.path.join(self._data_path, "train")
            self._val_path_base = os.path.join(self._data_path, "val")

            trainval_dict = {"train": {"path": self._train_path_base}, "val": {"path": self._val_path_base}}

            data_path = trainval_dict[self._phase]["path"]
            all_image_paths = DatasetBase.get_all_files_with_format_from_path(data_path, ".jpg")
            all_json_paths = DatasetBase.get_all_files_with_format_from_path(data_path, ".json")

            assert len(all_image_paths) == len(all_json_paths)

            self._image_paths = [os.path.join(data_path, elem) for elem in all_image_paths]
            self._targets = [self._load_one_json(os.path.join(data_path, elem)) for elem in all_json_paths]

    def _load_one_json(self, json_path):
        bboxes = []
        labels = []

        p_json = json.load(open(json_path, "r"))

        for obj in p_json["objects"]:
            bboxes.append(obj["boundingbox"])
            label_text = obj["label"]

            labels.append(0)

        return [bboxes, labels]
