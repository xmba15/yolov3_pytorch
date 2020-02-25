#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from .dataset_base import DatasetBase, DatasetConfigBase
from pycocotools.coco import COCO


class COCODatasetConfig(DatasetConfigBase):
    def __init__(self):
        super(COCODatasetConfig, self).__init__()

        self.CLASSES = [
            "person",
            "bicycle",
            "car",
            "motorbike",
            "aeroplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "sofa",
            "pottedplant",
            "bed",
            "diningtable",
            "toilet",
            "tvmonitor",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]

        self.COLORS = DatasetConfigBase.generate_color_chart(self.num_classes)


_coco_config = COCODatasetConfig()


class COCODataset(DatasetBase):
    __name__ = "coco_dataset"

    def __init__(
        self,
        data_path,
        data_path_suffix="coco_dataset",
        classes=_coco_config.CLASSES,
        colors=_coco_config.COLORS,
        phase="train",
        transform=None,
        shuffle=True,
        input_size=None,
        random_seed=2000,
        normalize_bbox=False,
        normalize_image=False,
        bbox_transformer=None,
    ):
        super(COCODataset, self).__init__(
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
        self._min_size = 1

        assert phase in ("train", "val")
        self._data_path = os.path.join(data_path, data_path_suffix)

        self._train_img_path = os.path.join(self._data_path, "train2017")
        self._val_img_path = os.path.join(self._data_path, "val2017")
        self._annotation_path = os.path.join(self._data_path, "annotations_trainval2017", "annotations")

        assert os.path.isdir(self._train_img_path)
        assert os.path.isdir(self._val_img_path)
        assert os.path.isdir(self._annotation_path)
        self._train_annotation_file = os.path.join(self._annotation_path, "instances_train2017.json")
        self._val_annotation_file = os.path.join(self._annotation_path, "instances_val2017.json")

        self._train_coco = COCO(self._train_annotation_file)
        self._val_coco = COCO(self._val_annotation_file)
        self._train_ids = self._train_coco.getImgIds()
        self._val_ids = self._train_coco.getImgIds()
        self._class_ids = sorted(self._train_coco.getCatIds())

        self._map_info = {
            "train": {"img_path": self._train_img_path, "coco": self._train_coco, "ids": self._train_ids,},
            "val": {"img_path": self._val_img_path, "coco": self._val_coco, "ids": self._val_ids,},
        }

        self._image_paths = []
        self._targets = []

        for idx in self._map_info[self._phase]["ids"]:
            anno_ids = self._map_info[self._phase]["coco"].getAnnIds(imgIds=[int(idx)], iscrowd=None)
            annotations = self._map_info[self._phase]["coco"].loadAnns(anno_ids)

            image_path = os.path.join(self._map_info[self._phase]["img_path"], "{:012}".format(idx) + ".jpg",)
            self._image_paths.append(image_path)

            cur_boxes = []
            cur_labels = []

            for anno in annotations:
                xmin, ymin, width, height = anno["bbox"]
                if width < self._min_size or height < self._min_size:
                    continue
                xmax = xmin + width
                ymax = ymin + height
                label = self._class_ids.index(anno["category_id"])
                cur_boxes.append([xmin, ymin, xmax, ymax])
                cur_labels.append(label)

            self._targets.append([cur_boxes, cur_labels])
