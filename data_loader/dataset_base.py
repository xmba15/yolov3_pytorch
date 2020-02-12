#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import tqdm
import multiprocessing as mp
from ctypes import c_int32
from abc import abstractmethod


_counter = mp.Value(c_int32)
_counter_lock = mp.Lock()


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
        multiscale=False,
        resize_after_batch_num=10,
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

        self._batch_count = 0
        self._multiscale = multiscale
        self._resize_after_batch_num = resize_after_batch_num

    @property
    def classes(self):
        return self._classes

    @property
    def colors(self):
        return self._colors

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

        return DatasetBase.visualize_one_image_util(image, self._classes, self._colors, all_bboxes, all_category_ids)

    @staticmethod
    def visualize_one_image_util(image, classes, colors, all_bboxes, all_category_ids):
        for (bbox, label) in zip(all_bboxes, all_category_ids):
            x_min, y_min, x_max, y_max = bbox

            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), colors[label], 2)

            label_text = classes[label]
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)

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

        o_bboxes = [DatasetBase.authentize_bbox(o_height, o_width, bbox) for bbox in o_bboxes]

        img = o_img
        bboxes = o_bboxes
        category_ids = o_category_ids
        height, width = o_height, o_width
        if self._transform:
            if isinstance(self._transform, list):
                for transform in self._transform:
                    img, bboxes, category_ids = transform(
                        image=img, bboxes=bboxes, labels=category_ids, phase=self._phase
                    )
            else:
                img, bboxes, category_ids = self._transform(
                    image=img, bboxes=bboxes, labels=category_ids, phase=self._phase
                )

            # use the height, width after transformation for normalization
            height, width, _ = img.shape

        # if number of boxes is 0, use original image
        # see data transform for more details

        if self._normalize_bbox:
            bboxes = [
                [float(bbox[0]) / width, float(bbox[1]) / height, float(bbox[2]) / width, float(bbox[3]) / height,]
                for bbox in bboxes
            ]

        bboxes = np.array(bboxes)
        category_ids = np.array(category_ids).reshape(-1, 1)
        targets = np.concatenate((bboxes, category_ids), axis=-1)

        return img, targets

    def _process_one_image(self, idx):
        abs_image_path = self._image_paths[idx]
        o_img = cv2.imread(abs_image_path)
        o_height, o_width, _ = o_img.shape

        o_bboxes, _ = self._targets[idx]

        o_bboxes = [DatasetBase.authentize_bbox(o_height, o_width, bbox) for bbox in o_bboxes]

        o_bboxes = np.array(o_bboxes)
        widths = (o_bboxes[:, 2] - o_bboxes[:, 0]) / o_width
        heights = (o_bboxes[:, 3] - o_bboxes[:, 1]) / o_height
        normalized_dimensions = [[w, h] for w, h in zip(widths, heights)]

        with _counter_lock:
            _counter.value += 1

        return normalized_dimensions

    def get_all_normalized_boxes(self, num_processes=mp.cpu_count()):
        import functools

        _process_len = self.__len__()

        pbar = tqdm.tqdm(total=_process_len)

        result_bboxes = None
        with mp.Pool(num_processes) as p:
            future = p.map_async(self._process_one_image, range(_process_len))
            while not future.ready():
                if _counter.value != 0:
                    with _counter_lock:
                        increment = _counter.value
                        _counter.value = 0
                    pbar.update(n=increment)

            result_bboxes = future.get()
            result_bboxes = functools.reduce(lambda x, y: x + y, result_bboxes)

        pbar.close()
        return np.array(result_bboxes)

    def _process_one_image_to_get_size(self, idx):
        abs_image_path = self._image_paths[idx]
        o_img = cv2.imread(abs_image_path)
        o_height, o_width, _ = o_img.shape
        o_size = o_height * o_width
        o_bboxes, _ = self._targets[idx]

        cur_sizes = []
        for (x1, y1, x2, y2) in o_bboxes:
            cur_sizes.append(np.sqrt((x2 - x1) * (y2 - y1) * 1.0 / o_size))

        return cur_sizes

    def size_distribution(self, num_processes=mp.cpu_count()):

        import functools

        _process_len = self.__len__()

        pbar = tqdm.tqdm(total=_process_len)

        result_bboxes = None
        with mp.Pool(num_processes) as p:
            future = p.map_async(self._process_one_image_to_get_size, range(_process_len))
            while not future.ready():
                if _counter.value != 0:
                    with _counter_lock:
                        increment = _counter.value
                        _counter.value = 0
                    pbar.update(n=increment)

            result_bboxes = future.get()
            result_bboxes = functools.reduce(lambda x, y: x + y, result_bboxes)

        pbar.close()
        return np.array(result_bboxes)

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
        return [int(c) if c.isdigit() else c.lower() for c in re.split(pattern, s)]

    @staticmethod
    def get_all_files_with_format_from_path(dir_path, suffix_format, use_human_sort=True):
        import os

        all_files = [elem for elem in os.listdir(dir_path) if elem.endswith(suffix_format)]
        all_files.sort(key=DatasetBase.human_sort)

        return all_files

    def od_collate_fn(self, batch):
        import torch
        import numpy as np

        def _xywh_to_cxcywh(bbox):
            bbox[..., 0] += bbox[..., 2] / 2
            bbox[..., 1] += bbox[..., 3] / 2
            return bbox

        def _xyxy_to_cxcywh(bbox):
            bbox[..., 2] -= bbox[..., 0]
            bbox[..., 3] -= bbox[..., 1]
            return _xywh_to_cxcywh(bbox)

        if (
            self._multiscale
            and (self._batch_count + 1) % self._resize_after_batch_num == 0
            and self._transform is not None
        ):
            if isinstance(self._transform, list):
                for i in range(len(self._transform)):
                    self._transform[i].update_size()
            else:
                self._transform.update_size()

        images = []
        labels = []
        lengths = []
        labels_with_tail = []
        max_num_obj = 0

        for image, label in batch:
            image = np.transpose(image, (2, 1, 0))
            image = np.expand_dims(image, axis=0)
            images.append(image)

            # xmin,ymin,xmax,ymax to xcenter,ycenter,width,height
            labels.append(_xyxy_to_cxcywh(label))

            length = label.shape[0]
            lengths.append(length)
            max_num_obj = max(max_num_obj, length)

        for label in labels:
            num_obj = label.shape[0]
            zero_tail = np.zeros((max_num_obj - num_obj, label.shape[1]), dtype=float)
            label_with_tail = np.concatenate([label, zero_tail], axis=0)
            labels_with_tail.append(torch.FloatTensor(label_with_tail))

        images = np.concatenate(images, axis=0)

        image_tensor = torch.FloatTensor(images)
        label_tensor = torch.stack(labels_with_tail)
        length_tensor = torch.tensor(lengths)

        self._batch_count += 1

        return image_tensor, label_tensor, length_tensor
