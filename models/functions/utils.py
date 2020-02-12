#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch
from .new_types import BboxType


__all__ = ["bbox_iou", "nms", "transform_bbox"]


def bbox_iou(
    bboxes1: torch.Tensor, bboxes2: torch.Tensor, bbox_mode: BboxType = BboxType.XYXY, epsilon: float = 1e-16
) -> torch.Tensor:
    """
    Args:
    bboxes1: (num_boxes_1, 4)
    bboxes2: (num_boxes_2, 4)

    Return:
    ious: (num_boxes_1, num_boxes_2)
    """
    if bbox_mode == BboxType.XYXY:
        b1_x1, b1_y1, b1_x2, b1_y2 = (
            bboxes1[..., 0],
            bboxes1[..., 1],
            bboxes1[..., 2],
            bboxes1[..., 3],
        )
        b2_x1, b2_y1, b2_x2, b2_y2 = (
            bboxes2[..., 0],
            bboxes2[..., 1],
            bboxes2[..., 2],
            bboxes2[..., 3],
        )
    elif bbox_mode == BboxType.CXCYWH:
        b1_x1, b1_x2 = bboxes1[..., 0] - bboxes1[..., 2] / 2, bboxes1[..., 0] + bboxes1[..., 2] / 2
        b1_y1, b1_y2 = bboxes1[..., 1] - bboxes1[..., 3] / 2, bboxes1[..., 1] + bboxes1[..., 3] / 2
        b2_x1, b2_x2 = bboxes2[..., 0] - bboxes2[..., 2] / 2, bboxes2[..., 0] + bboxes2[..., 2] / 2
        b2_y1, b2_y2 = bboxes2[..., 1] - bboxes2[..., 3] / 2, bboxes2[..., 1] + bboxes2[..., 3] / 2
    elif bbox_mode == BboxType.XYWH:
        b1_x1, b1_y1 = bboxes1[..., 0], bboxes1[..., 1]
        b2_x1, b2_y1 = bboxes2[..., 0], bboxes2[..., 1]
        b1_x2, b1_y2 = bboxes1[..., 0] + bboxes1[..., 2], bboxes1[..., 1] + bboxes1[..., 3]
        b2_x2, b2_y2 = bboxes2[..., 0] + bboxes2[..., 2], bboxes2[..., 1] + bboxes2[..., 3]
    else:
        raise Exception("not supported bbox type\n")

    num_b1 = bboxes1.shape[0]
    num_b2 = bboxes2.shape[0]

    inter_x1 = torch.max(b1_x1.unsqueeze(1).repeat(1, num_b2), b2_x1)
    inter_y1 = torch.max(b1_y1.unsqueeze(1).repeat(1, num_b2), b2_y1)
    inter_x2 = torch.min(b1_x2.unsqueeze(1).repeat(1, num_b2), b2_x2)
    inter_y2 = torch.min(b1_y2.unsqueeze(1).repeat(1, num_b2), b2_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area.unsqueeze(1).repeat(1, num_b2) + b2_area.unsqueeze(0).repeat(num_b1, 1) - inter_area + epsilon

    iou = inter_area / union_area
    return iou


def nms(
    bboxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    num_classes: int,
    conf_thresh: float = 0.8,
    nms_thresh: float = 0.5,
    bbox_mode: BboxType = BboxType.CXCYWH,
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Args:
    bboxes: The location predictions for the img, Shape: [num_anchors,4].
    scores: The class prediction scores for the img, Shape:[num_anchors].
    classes: The label (non-one-hot) representation of the classes of the objects,
        Shape: [num_anchors].
    num_classes: number of classes
    conf_thresh: threshold where all the detections below this value will be ignored.
    nms_thresh: overlap thresh for suppressing unnecessary boxes.

    Return:
    (bboxes, scores, classes) after nms suppression with bboxes in BboxType.XYXY box mode
    """

    assert bboxes.shape[0] == scores.shape[0] == classes.shape[0]
    assert conf_thresh > 0

    num_anchors = bboxes.shape[0]

    if num_anchors == 0:
        return bboxes, scores, classes

    conf_index = torch.nonzero(torch.ge(scores, conf_thresh)).squeeze()
    bboxes = bboxes.index_select(0, conf_index)
    scores = scores.index_select(0, conf_index)
    classes = classes.index_select(0, conf_index)

    grouped_indices = _group_same_class_object(classes, one_hot=False, num_classes=num_classes)
    selected_indices_final = []

    for class_id, member_idx in enumerate(grouped_indices):
        member_idx_tensor = bboxes.new_tensor(member_idx, dtype=torch.long)
        bboxes_one_class = bboxes.index_select(dim=0, index=member_idx_tensor)
        scores_one_class = scores.index_select(dim=0, index=member_idx_tensor)
        scores_one_class, sorted_indices = torch.sort(scores_one_class, descending=False)

        selected_indices = []

        while sorted_indices.size(0) != 0:
            picked_index = sorted_indices[-1]
            selected_indices.append(picked_index)
            picked_bbox = bboxes_one_class[picked_index]

            picked_bbox.unsqueeze_(dim=0)

            ious = bbox_iou(picked_bbox, bboxes_one_class[sorted_indices[:-1]], bbox_mode=bbox_mode)
            ious.squeeze_(dim=0)

            under_indices = torch.nonzero(ious <= nms_thresh).squeeze()
            sorted_indices = sorted_indices.index_select(dim=0, index=under_indices)

        selected_indices_final.extend([member_idx[i] for i in selected_indices])

    selected_indices_final = bboxes.new_tensor(selected_indices_final, dtype=torch.long)
    bboxes_result = bboxes.index_select(dim=0, index=selected_indices_final)
    scores_result = scores.index_select(dim=0, index=selected_indices_final)
    classes_result = classes.index_select(dim=0, index=selected_indices_final)

    return transform_bbox(bboxes_result, orig_mode=bbox_mode, target_mode=BboxType.XYXY), scores_result, classes_result


def transform_bbox(
    bboxes: torch.Tensor, orig_mode: BboxType = BboxType.CXCYWH, target_mode: BboxType = BboxType.XYXY
) -> torch.Tensor:
    assert orig_mode != target_mode
    assert bboxes.shape[1] == 4

    if orig_mode == BboxType.CXCYWH:
        if target_mode == BboxType.XYXY:
            return torch.cat((bboxes[:, :2] - bboxes[:, 2:] / 2, bboxes[:, :2] + bboxes[:, 2:] / 2), dim=-1,)
        elif target_mode == BboxType.XYWH:
            return torch.cat((bboxes[:, :2] - bboxes[:, 2:] / 2, bboxes[:, :2]), dim=-1,)
        else:
            raise Exception("not supported conversion\n")
    elif orig_mode == BboxType.XYWH:
        if target_mode == BboxType.XYXY:
            return torch.cat((bboxes[:, :2], bboxes[:, 2:] + bboxes[:, :2]), dim=-1,)
        elif target_mode == BboxType.CXCYWH:
            return torch.cat((bboxes[:, :2] + bboxes[:, 2:] / 2, bboxes[:, 2:]), dim=-1,)
        else:
            raise Exception("not supported conversion\n")
    elif orig_mode == BboxType.XYXY:
        if target_mode == BboxType.CXCYWH:
            return torch.cat(((bboxes[:, :2] + bboxes[:, 2:]) / 2, bboxes[:, 2:] - bboxes[:, :2]), dim=-1,)
        if target_mode == BboxType.XYWH:
            return torch.cat((bboxes[:, :2], bboxes[:, 2:] - bboxes[:, :2]), dim=-1,)
        else:
            raise Exception("not supported conversion\n")
    else:
        raise Exception("not supported original bbox mode\n")


def _group_same_class_object(obj_classes: torch.Tensor, one_hot: bool = True, num_classes: int = -1):
    """
    Given a list of class results, group the object with the same class into a list.
    Returns a list with the length of num_classes, where each bucket has the objects with the same class.

    Args:
    obj_classes: The representation of classes of object.
         It can be either one-hot or label (non-one-hot).
         If it is one-hot, the shape should be: [num_objects, num_classes]
         If it is label (non-non-hot), the shape should be: [num_objects, ]
    one_hot: A flag telling the function whether obj_classes is one-hot representation.
    num_classes: The max number of classes if obj_classes is represented as non-one-hot format.

    Returns:
    a list of of a list, where for the i-th list,
    the elements in such list represent the indices of the objects in class i.
    """

    if one_hot:
        num_classes = obj_classes.shape[-1]
    else:
        assert num_classes != -1
    grouped_index = [[] for _ in range(num_classes)]
    if one_hot:
        for idx, class_one_hot in enumerate(obj_classes):
            grouped_index[torch.argmax(class_one_hot)].append(idx)
    else:
        for idx, obj_class_ in enumerate(obj_classes):
            grouped_index[obj_class_].append(idx)

    return grouped_index
