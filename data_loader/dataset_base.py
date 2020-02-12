#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import tqdm
import multiprocessing
from abc import abstractmethod


_result_bboxes = []


class DatasetConfigBase(object):
    def __init__(self):
        self.CLASSES = []

    @property
    def num_classes(self):
        return len(self.CLASSES)

    @staticmethod
    def generate_color_chart(num_classes, seed=3700):
        assert num_classes > 0
        np.random.seed(seed)

        colors = np.random.randint(0, 255, size=(num_classes, 3), dtype="uint8")
        colors = np.vstack([colors]).astype("uint8")
        colors = [tuple(color) for color in list(colors)]
        colors = [tuple(int(e) for e in color) for color in colors]

        return colors


class DatasetBase(object):
    def __init__(
        self,
        data_path,
        classes,
        colors,
        phase="train",
        transform=None,
        shuffle=True,
        normalize_bbox=False,
        bbox_transformer=None,
    ):
        super(DatasetBase, self).__init__()
        assert os.path.isdir(data_path)
        assert phase in ("train", "val", "test")

        self._data_path = data_path
        self._classes = classes
        self._colors = colors
        self._phase = phase
        self._transform = transform
        self._shuffle = shuffle
        self._normalize_bbox = normalize_bbox
        self._bbox_transformer = bbox_transformer
        self._image_paths = []
        self._targets = []

    @property
    def num_classes(self):
        return len(self._classes)

    def __getitem__(self, idx):
        """
        X: np.array (batch_size, height, width, channel)
        y: list of length batch size
        y: (batch_size, [[number of bboxes, x_min, y_min, x_max, y_max, labels]])
        """
        assert idx < self.__len__()

        image, targets = self._data_generation(idx)

        if self._bbox_transformer is not None:
            targets = self._bbox_transformer(targets)

        return image, targets

    def __len__(self):
        return len(self._image_paths)

    def visualize_one_image(self, idx):
        assert not self._normalize_bbox
        assert idx < self.__len__()
        image, targets = self.__getitem__(idx)

        all_bboxes = targets[:, :-1]
        all_category_ids = targets[:, -1]

        all_bboxes = all_bboxes.astype(np.int64)
        all_category_ids = all_category_ids.astype(np.int64)

        return DatasetBase.visualize_one_image_util(
            image, self._classes, self._colors, all_bboxes, all_category_ids
        )

    @staticmethod
    def visualize_one_image_util(
        image, classes, colors, all_bboxes, all_category_ids
    ):
        for (bbox, label) in zip(all_bboxes, all_category_ids):
            x_min, y_min, x_max, y_max = bbox

            cv2.rectangle(
                image, (x_min, y_min), (x_max, y_max), colors[label], 2
            )

            label_text = classes[label]
            label_size = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1
            )

            cv2.rectangle(
                image,
                (x_min, y_min),
                (x_min + label_size[0][0], y_min + int(1.3 * label_size[0][1])),
                colors[label],
                -1,
            )
            cv2.putText(
                image,
                label_text,
                org=(x_min, y_min + label_size[0][1]),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.35,
                color=(255, 255, 255),
                lineType=cv2.LINE_AA,
            )

        return image

    def _data_generation(self, idx):
        abs_image_path = self._image_paths[idx]
        o_img = cv2.imread(abs_image_path)
        o_height, o_width, _ = o_img.shape

        o_bboxes, o_category_ids = self._targets[idx]

        o_bboxes = [
            DatasetBase.authentize_bbox(o_height, o_width, bbox)
            for bbox in o_bboxes
        ]

        img = o_img
        bboxes = o_bboxes
        category_ids = o_category_ids
        height, width = o_height, o_width
        if self._transform:
            if isinstance(self._transform, list):
                for transform in self._transform:
                    img, bboxes, category_ids = transform(
                        img, bboxes, category_ids, phase=self._phase
                    )
            else:
                img, bboxes, category_ids = self._transform(
                    img, bboxes, category_ids, phase=self._phase
                )

            # use the height, width after transformation for normalization
            height, width, _ = img.shape

        # if number of boxes is 0, use original image
        # see data transform for more details

        if self._normalize_bbox:
            bboxes = [
                [
                    float(bbox[0]) / width,
                    float(bbox[1]) / height,
                    float(bbox[2]) / width,
                    float(bbox[3]) / height,
                ]
                for bbox in bboxes
            ]

        bboxes = np.array(bboxes)
        category_ids = np.array(category_ids).reshape(-1, 1)
        targets = np.concatenate((bboxes, category_ids), axis=-1)

        return img, targets

    def _process_one_image(self, idx):
        global _result_bboxes
        abs_image_path = self._image_paths[idx]
        o_img = cv2.imread(abs_image_path)
        o_height, o_width, _ = o_img.shape

        o_bboxes, _ = self._targets[idx]

        o_bboxes = [
            DatasetBase.authentize_bbox(o_height, o_width, bbox)
            for bbox in o_bboxes
        ]

        o_bboxes = np.array(o_bboxes)
        widths = (o_bboxes[:, 2] - o_bboxes[:, 0]) / o_width
        heights = (o_bboxes[:, 3] - o_bboxes[:, 1]) / o_height
        normalized_dimensions = [[w, h] for w, h in zip(widths, heights)]

        _result_bboxes += normalized_dimensions

    def get_all_normalized_boxes(
        self, num_processes=multiprocessing.cpu_count()
    ):
        global _result_bboxes

        with multiprocessing.Pool(num_processes) as p:
            r = list(
                tqdm.tqdm(
                    p.imap(self._process_one_image, range(self.__len__())),
                    total=self.__len__(),
                )
            )

        return np.array(_result_bboxes)

    @staticmethod
    def authentize_bbox(o_height, o_width, bbox):
        bbox_type = type(bbox)

        x_min, y_min, x_max, y_max = bbox
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if y_min > y_max:
            y_min, y_max = y_max, y_min

        x_min = max(x_min, 0)
        x_max = min(x_max, o_width)
        y_min = max(y_min, 0)
        y_max = min(y_max, o_height)

        return bbox_type([x_min, y_min, x_max, y_max])

    @staticmethod
    def color_to_color_idx_dict(colors):
        color_idx_dict = {}

        for i, color in enumerate(colors):
            color_idx_dict[color] = i

        return color_idx_dict

    @staticmethod
    def class_to_class_idx_dict(classes):
        class_idx_dict = {}

        for i, class_name in enumerate(classes):
            class_idx_dict[class_name] = i

        return class_idx_dict

    @staticmethod
    def human_sort(s):
        """Sort list the way humans do
        """
        import re

        pattern = r"([0-9]+)"
        return [
            int(c) if c.isdigit() else c.lower() for c in re.split(pattern, s)
        ]
