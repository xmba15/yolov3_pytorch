#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .kmeans_bboxes import kmeans_bboxes


def estimate_priors_sizes(dataset, k=5):
    bboxes = dataset.get_all_normalized_boxes()
    return kmeans_bboxes(bboxes, k)
