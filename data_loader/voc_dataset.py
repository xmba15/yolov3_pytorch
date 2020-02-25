#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from .dataset_base import DatasetBase, DatasetConfigBase


class VOCDatasetConfig(DatasetConfigBase):
    def __init__(self):
        super(VOCDatasetConfig, self).__init__()

        self.CLASSES = [
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]

        self.COLORS = DatasetConfigBase.generate_color_chart(self.num_classes)


_voc_config = VOCDatasetConfig()


class VOCDataset(DatasetBase):
    __name__ = "voc_dataset"

    def __init__(
        self,
        data_path,
        classes=_voc_config.CLASSES,
        colors=_voc_config.COLORS,
        phase="train",
        transform=None,
        shuffle=True,
        input_size=None,
        random_seed=2000,
        normalize_bbox=False,
        normalize_image=False,
        bbox_transformer=None,
    ):
        super(VOCDataset, self).__init__(
            data_path,
            classes=classes,
            colors=colors,
            phase=phase,
            transform=transform,
            shuffle=shuffle,
            normalize_bbox=normalize_bbox,
            bbox_transformer=bbox_transformer,
        )

        self._input_size = input_size
        self._normalize_image = normalize_image

        assert phase in ("train", "val")
        self._data_path = os.path.join(self._data_path, "voc/VOCdevkit/VOC2012")
        self._image_path_file = os.path.join(self._data_path, "ImageSets/Main/{}.txt".format(self._phase),)

        lines = [line.rstrip("\n") for line in open(self._image_path_file)]

        image_path_base = os.path.join(self._data_path, "JPEGImages")
        anno_path_base = os.path.join(self._data_path, "Annotations")

        self._image_paths = []
        anno_paths = []

        for line in lines:
            self._image_paths.append(os.path.join(image_path_base, "{}.jpg".format(line)))
            anno_paths.append(os.path.join(anno_path_base, "{}.xml".format(line)))

        self._targets = [self._parse_one_xml(xml_path) for xml_path in anno_paths]

        np.random.seed(random_seed)

    def _parse_one_xml(self, xml_path):
        bboxes = []
        labels = []
        xml = ET.parse(xml_path).getroot()

        for obj in xml.iter("object"):
            difficult = int(obj.find("difficult").text)
            if difficult == 1:
                continue

            bndbox = []

            name = obj.find("name").text.lower().strip()
            bbox = obj.find("bndbox")

            pts = ["xmin", "ymin", "xmax", "ymax"]

            for pt in pts:
                cur_pixel = int(bbox.find(pt).text) - 1
                bndbox.append(cur_pixel)

            label_idx = self._classes.index(name)

            bboxes.append(bndbox)
            labels.append(label_idx)

        return [bboxes, labels]
