#!/usr/bin/env python
# -*- coding: utf-8 -*-
from albumentations import *
from albumentations.pytorch import ToTensor
import random

_SEED = 100
random.seed(_SEED)


class DataTransformBase(object):
    def __init__(
        self,
        transforms=[HorizontalFlip(p=0.5), GaussNoise(p=0.5), RandomBrightnessContrast(p=0.5),],
        input_size=None,
        normalize=False,
    ):
        self._input_size = input_size
        if self._input_size is not None:
            height, width = self._input_size
            self._height_offset = height // 32 - 8
            self._width_offset = width // 32 - 8
            assert self._height_offset > 0 and self._width_offset > 0

        self._normalize = normalize

        self._transform_dict = {"train": {}, "val": {}, "test": None}

        self._transform_dict["train"]["normal"] = transforms
        self._transform_dict["val"]["normal"] = []

        self._bbox_params = BboxParams(
            format="pascal_voc", min_area=0.001, min_visibility=0.001, label_fields=["category_ids"],
        )
        self._initialize_transform_dict()

    def _get_all_transforms_of_phase(self, phase):
        assert phase in ("train", "val")
        cur_transform = []
        cur_transform.extend(self._transform_dict[phase]["normal"])
        cur_transform.append(self._transform_dict[phase]["resize"])
        cur_transform.append(self._transform_dict[phase]["normalize"])

        return cur_transform

    def _initialize_transform_dict(self):
        if self._input_size is not None:
            height, width = self._input_size
            self._transform_dict["train"]["resize"] = Resize(height, width, always_apply=True)
            self._transform_dict["val"]["resize"] = Resize(height, width, always_apply=True)

        if self._normalize:
            self._transform_dict["train"]["normalize"] = Normalize(always_apply=True)
            self._transform_dict["val"]["normalize"] = Normalize(always_apply=True)
        else:
            self._transform_dict["train"]["normalize"] = ToTensor()
            self._transform_dict["val"]["normalize"] = ToTensor()

        self._transform_dict["train"]["all"] = self._get_all_transforms_of_phase("train")
        self._transform_dict["val"]["all"] = self._get_all_transforms_of_phase("val")

        self._transform_dict["test"] = self._transform_dict["val"]["all"]

    def update_size(self):
        if self._input_size is not None:
            random_offset = random.randint(0, 9)
            new_height = (random_offset + self._height_offset) * 32
            new_width = (random_offset + self._width_offset) * 32

            self._transform_dict["train"]["resize"] = Resize(new_height, new_width, always_apply=True)
            self._transform_dict["train"]["all"] = self._get_all_transforms_of_phase("train")

    def __call__(self, image, bboxes=None, labels=None, phase=None):
        if phase is None:
            transformer = Compose(self._transform_dict["test"])
            return transformer(image=image)

        assert phase in ("train", "val")
        assert bboxes is not None
        assert labels is not None

        transformed_image = image
        transformed_bboxes = bboxes
        transformed_category_ids = labels
        for transform in self._transform_dict[phase]["all"]:
            annotations = {
                "image": transformed_image,
                "bboxes": transformed_bboxes,
                "category_ids": transformed_category_ids,
            }
            transformer = Compose([transform], bbox_params=self._bbox_params)
            augmented = transformer(**annotations)

            while len(augmented["bboxes"]) == 0:
                augmented = transformer(**annotations)

            transformed_image = augmented["image"]
            transformed_bboxes = augmented["bboxes"]
            transformed_category_ids = augmented["category_ids"]

        if not self._normalize:
            transformed_image = transformed_image.permute(2, 1, 0)

        return (transformed_image, transformed_bboxes, transformed_category_ids)
