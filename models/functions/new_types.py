#!/usr/bin/env python
# -*- coding: utf-8 -*-
import enum


__all__ = ["BboxType"]


class BboxType(enum.Enum):
    """
    XYWH: xmin, ymin, width, height
    CXCYWH: xcenter, ycenter, width, height
    XYXY: xmin, ymin, xmax, ymax
    """

    XYWH = 0
    CXCYWH = 1
    XYXY = 2
