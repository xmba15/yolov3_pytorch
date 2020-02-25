#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F


__all__ = ["Mish"]


@torch.jit.script
def _mish(input):
    """
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    """
    return input * torch.tanh(F.softplus(input))


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return _mish(input)
