#!/usr/bin/env python
import os
import cv2
import random
import numpy as np
import json
from .dataset_base import DatasetBase, DatasetConfigBase


class WheatDatasetConfig(DatasetConfigBase):
    def __init__(self):
        super(WheatDatasetConfig, self).__init__()

        self.CLASSES = [
            "wheat",
        ]

        self.COLORS = DatasetConfigBase.generate_color_chart(self.num_classes)


_wheat_config = WheatDatasetConfig()


class WheatDataset(DatasetBase):
    __name__ = "wheat_dataset"

    def __init__(
        self,
        data_path,
        classes=_wheat_config.CLASSES,
        colors=_wheat_config.COLORS,
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
        super(WheatDataset, self).__init__(
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

        self._wheat_data_path = os.path.join(data_path, "global-wheat-detection")
        assert os.path.isdir(self._wheat_data_path)
        self._train_csv_path = os.path.join(self._wheat_data_path, "train.csv")
        self._image_prefix_path = os.path.join(self._wheat_data_path, "train")

        self._phase = phase
        self._transform = transform

        self._image_paths, self._targets = self._process_train_csv(self._train_csv_path)
        self._image_paths = [os.path.join(self._image_prefix_path, elem) + ".jpg" for elem in self._image_paths]

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

    def _process_train_csv(self, train_csv_path):
        image_dict = {}

        import dask.dataframe as dd

        df = dd.read_csv(train_csv_path)
        for idx, row in df.iterrows():
            image_id = row["image_id"]
            if image_id not in image_dict.keys():
                image_dict[image_id] = [[], []]

            source = row["source"]
            width = row["width"]
            height = row["height"]

            bbox = row["bbox"].strip("][").split(", ")
            bbox = [float(elem) for elem in bbox]

            xmin, ymin, width, height = bbox
            xmax = xmin + width
            ymax = ymin + height

            bbox = [int(xmin), int(ymin), int(xmax), int(ymax)]

            image_dict[image_id][0].append(bbox)
            image_dict[image_id][1].append(0)

        return image_dict.keys(), image_dict.values()
