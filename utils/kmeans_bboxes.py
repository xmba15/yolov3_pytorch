#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ref: https://lars76.github.io/object-detection/k-means-anchor-boxes/
import numpy as np


def jaccard(bboxes, clusters):
    x = np.minimum(clusters[:, 0], bboxes[0])
    y = np.minimum(clusters[:, 1], bboxes[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = bboxes[0] * bboxes[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou = intersection / (box_area + cluster_area - intersection)

    return iou


def avg_iou(bboxes, clusters):
    return np.mean([np.max(jaccard(bboxes[i], clusters)) for i in range(bboxes.shape[0])])


def kmeans_bboxes(bboxes, k=5, metric_dist=np.median, seed=100):
    assert k >= 2
    rows = bboxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed(seed)

    clusters = bboxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - jaccard(bboxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = metric_dist(bboxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters


def estimate_avg_ious(bboxes, k=5, metric_dist=np.median, seed=100):
    assert k >= 2

    num_vals = k - 1
    avg_ious = np.empty((num_vals,), dtype=float)
    for idx, k in enumerate(tqdm.tqdm(range(2, k + 1))):
        clusters = kmeans_bboxes(bboxes, k, metric_dist, seed)
        avg_ious[idx] = avg_iou(bboxes, clusters)

    return avg_ious
