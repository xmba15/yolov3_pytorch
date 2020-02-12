#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import random
from .dataset_base import DatasetBase, DatasetConfigBase


class UdacityDatasetConfig(DatasetConfigBase):
    def __init__(self):
        super(UdacityDatasetConfig, self).__init__()

        self.CLASSES = ["car", "truck", "pedestrian", "biker", "trafficLight"]

        self.COLORS = DatasetConfigBase.generate_color_chart(self.num_classes)


_udacity_config = UdacityDatasetConfig()


class UdacityDataset(DatasetBase):
    __name__ = "udacity_dataset"

    def __init__(
        self,
        data_path,
        classes=_udacity_config.CLASSES,
        colors=_udacity_config.COLORS,
        phase="train",
        transform=None,
        shuffle=True,
        random_seed=2000,
        normalize_bbox=False,
        bbox_transformer=None,
        train_val_ratio=0.9,
    ):
        super(UdacityDataset, self).__init__(
            data_path,
            classes=classes,
            colors=colors,
            phase=phase,
            transform=transform,
            shuffle=shuffle,
            normalize_bbox=normalize_bbox,
            bbox_transformer=bbox_transformer,
        )

        assert phase in ("train", "val")

        assert os.path.isdir(data_path)
        self._data_path = os.path.join(data_path, "udacity/object-dataset")
        assert os.path.isdir(self._data_path)

        self._annotation_file = os.path.join(self._data_path, "labels.csv")
        lines = [line.rstrip("\n") for line in open(self._annotation_file, "r")]
        lines = [line.split(" ") for line in lines]
        image_dict = {}
        class_idx_dict = self.class_to_class_idx_dict(self._classes)

        for line in lines:
            if line[0] not in image_dict.keys():
                image_dict[line[0]] = [[], []]

            image_dict[line[0]][0].append([int(e) for e in line[1:5]])
            label_name = line[6][1:][:-1]
            image_dict[line[0]][1].append(class_idx_dict[label_name])

        self._image_paths = image_dict.keys()
        self._image_paths = [os.path.join(self._data_path, elem) for elem in self._image_paths]
        self._targets = image_dict.values()

        zipped = list(zip(self._image_paths, self._targets))
        random.seed(random_seed)
        random.shuffle(zipped)
        self._image_paths, self._targets = zip(*zipped)

        train_len = int(train_val_ratio * len(self._image_paths))
        if self._phase == "train":
            self._image_paths = self._image_paths[:train_len]
            self._targets = self._targets[:train_len]
        else:
            self._image_paths = self._image_paths[train_len:]
            self._targets = self._targets[train_len:]
