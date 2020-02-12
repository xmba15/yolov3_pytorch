#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from models.functions import Mish


__all__ = ["ConvBnAct", "Darknet", "PreDetectionConvGroup", "UpsampleGroup"]


def _activation_func(activation):
    import sys
    import copy

    try:
        return copy.deepcopy(
            nn.ModuleDict(
                [
                    ["relu", nn.ReLU(inplace=True)],
                    ["leaky_relu", nn.LeakyReLU(negative_slope=0.1, inplace=True)],
                    ["selu", nn.SELU(inplace=True)],
                    ["mish", Mish()],
                    ["identiy", nn.Identity()],
                ]
            )[activation]
        )
    except Exception as e:
        print("no activation {}".format(activation))
        sys.exit(-1)


class ConvBnAct(nn.Module):
    def __init__(
        self, nin, nout, ks, s=1, pad="SAME", padding=0, use_bn=True, act="mish",
    ):
        super(ConvBnAct, self).__init__()

        self.use_bn = use_bn
        self.bn = None
        if use_bn:
            self.bn = nn.BatchNorm2d(nout)

        if pad == "SAME":
            padding = (ks - 1) // 2

        self.conv = nn.Conv2d(
            in_channels=nin, out_channels=nout, kernel_size=ks, stride=s, padding=padding, bias=not use_bn
        )
        self.activation = _activation_func(act)

    def forward(self, x):
        out = self.conv(x)
        if self.use_bn:
            out = self.bn(out)

        return self.activation(out)


class ResResidualBlock(nn.Module):
    def __init__(self, nin):
        super(ResResidualBlock, self).__init__()
        self.conv1 = ConvBnAct(nin, nin // 2, ks=1)
        self.conv2 = ConvBnAct(nin // 2, nin, ks=3)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


def _map_2_cfgdict(module_list):
    from collections import OrderedDict

    idx = 0
    mdict = OrderedDict()
    for i, m in enumerate(module_list):
        if isinstance(m, ResResidualBlock):
            mdict[idx] = None
            mdict[idx + 1] = None
            idx += 2
        mdict[idx] = i
        idx += 1
    return mdict


def _make_res_stack(nin, num_blk):
    return nn.ModuleList([ConvBnAct(nin, nin * 2, 3, s=2)] + [ResResidualBlock(nin * 2) for n in range(num_blk)])


class Darknet(nn.Module):
    def __init__(self, blk_list, nout=32):
        super(Darknet, self).__init__()

        self.module_list = nn.ModuleList()
        self.module_list += [ConvBnAct(3, nout, 3)]
        for i, nb in enumerate(blk_list):
            self.module_list += _make_res_stack(nout * (2 ** i), nb)

        self.map2yolocfg = _map_2_cfgdict(self.module_list)
        self.cached_out_dict = dict()

    def forward(self, x):
        for i, m in enumerate(self.module_list):
            x = m(x)
            if i in self.cached_out_dict:
                self.cached_out_dict[i] = x
        return x

    # mode - normal  -- direct index to module_list
    #     - yolocfg -- index follow the sequences of the cfg file from https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
    def add_cached_out(self, idx, mode="yolocfg"):
        if mode == "yolocfg":
            idxs = self.map2yolocfg[idx]
        self.cached_out_dict[idxs] = None

    def get_cached_out(self, idx, mode="yolocfg"):
        if mode == "yolocfg":
            idxs = self.map2yolocfg[idx]
        return self.cached_out_dict[idxs]

    def load_weight(self, weights_path):
        wm = WeightLoader(self)
        wm.load_weight(weights_path)


class PreDetectionConvGroup(nn.Module):
    def __init__(self, nin, nout, num_classes, num_anchors, num_conv=3):
        super(PreDetectionConvGroup, self).__init__()
        self.module_list = nn.ModuleList()

        for i in range(num_conv):
            self.module_list += [ConvBnAct(nin, nout, ks=1)]
            self.module_list += [ConvBnAct(nout, nout * 2, ks=3)]
            if i == 0:
                nin = nout * 2

        self.module_list += [nn.Conv2d(nin, (num_classes + 5) * num_anchors, 1)]
        self.map2yolocfg = _map_2_cfgdict(self.module_list)
        self.cached_out_dict = dict()

    def forward(self, x):
        for i, m in enumerate(self.module_list):
            x = m(x)
            if i in self.cached_out_dict:
                self.cached_out_dict[i] = x
        return x

    def add_cached_out(self, idx, mode="yolocfg"):
        if mode == "yolocfg":
            idx = self.get_idx_from_yolo_idx(idx)
        elif idx < 0:
            idx = len(self.module_list) - idx

        self.cached_out_dict[idx] = None

    def get_cached_out(self, idx, mode="yolocfg"):
        if mode == "yolocfg":
            idx = self.get_idx_from_yolo_idx(idx)
        elif idx < 0:
            idx = len(self.module_list) - idx
        return self.cached_out_dict[idx]

    def get_idx_from_yolo_idx(self, idx):
        if idx < 0:
            return len(self.map2yolocfg) + idx
        else:
            return self.map2yolocfg[idx]


class UpsampleGroup(nn.Module):
    def __init__(self, nin):
        super(UpsampleGroup, self).__init__()
        self.conv = ConvBnAct(nin, nin // 2, ks=1)

    def forward(self, route_head, route_tail):
        out = self.conv(route_head)
        out = nn.functional.interpolate(out, scale_factor=2, mode="nearest")
        return torch.cat((out, route_tail), 1)


class WeightLoader:
    def __init__(self, model):
        super(WeightLoader, self).__init__()
        self._conv_list = self._find_conv_layers(model)

    def load_weight(self, weight_path):
        ptr = 0
        weights = self._read_file(weight_path)
        for m in self._conv_list:
            if type(m) == ConvBnAct:
                ptr = self._load_conv_bn_relu(m, weights, ptr)
            elif type(m) == nn.Conv2d:
                ptr = self._load_conv2d(m, weights, ptr)
        return ptr

    def _read_file(self, file):
        import numpy as np

        with open(file, "rb") as fp:
            header = np.fromfile(fp, dtype=np.int32, count=5)
            self.header = torch.from_numpy(header)
            self.seen = self.header[3]
            weights = np.fromfile(fp, dtype=np.float32)
        return weights

    def _copy_weight_to_model_parameters(self, param, weights, ptr):
        num_el = param.numel()
        param.data.copy_(torch.from_numpy(weights[ptr : ptr + num_el]).view_as(param.data))
        return ptr + num_el

    def _load_conv_bn_relu(self, m, weights, ptr):
        ptr = self._copy_weight_to_model_parameters(m.bn.bias, weights, ptr)
        ptr = self._copy_weight_to_model_parameters(m.bn.weight, weights, ptr)
        ptr = self._copy_weight_to_model_parameters(m.bn.running_mean, weights, ptr)
        ptr = self._copy_weight_to_model_parameters(m.bn.running_var, weights, ptr)
        ptr = self._copy_weight_to_model_parameters(m.conv.weight, weights, ptr)
        return ptr

    def _load_conv2d(self, m, weights, ptr):
        ptr = self._copy_weight_to_model_parameters(m.bias, weights, ptr)
        ptr = self._copy_weight_to_model_parameters(m.weight, weights, ptr)
        return ptr

    def _find_conv_layers(self, mod):
        module_list = []
        for m in mod.children():
            if type(m) == ConvBnAct:
                module_list += [m]
            elif type(m) == nn.Conv2d:
                module_list += [m]
            elif isinstance(m, (nn.ModuleList, nn.Module)):
                module_list += self._find_conv_layers(m)
            elif type(m) == ResResidualBlock:
                module_list += self._find_conv_layers(m)
        return module_list
