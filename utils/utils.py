#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import numpy as np


__all__ = ["inf_loop"]


def inf_loop(data_loader):
    from itertools import repeat

    for loader in repeat(data_loader):
        yield from loader
