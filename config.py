#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import logging
from dataset_params import dataset_params
from data_loader import BirdDataset, UdacityDataset, VOCDataset, COCODataset, SwitchDataset, WheatDataset


_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


class Config(object):
    DATASETS = {
        "bird_dataset": BirdDataset,
        "udacity_dataset": UdacityDataset,
        "voc_dataset": VOCDataset,
        "coco_dataset": COCODataset,
        "switch_dataset": SwitchDataset,
        "wheat_dataset": WheatDataset,
    }

    DATASET_PARAMS = dataset_params

    def __init__(self):
        self.CURRENT_DIR = _CURRENT_DIR

        self.DATA_PATH = os.path.abspath(os.path.join(_CURRENT_DIR, "data"))

        self.SAVED_MODEL_PATH = os.path.join(self.CURRENT_DIR, "saved_models")
        if not os.path.isdir(self.SAVED_MODEL_PATH):
            os.system("mkdir -p {}".format(self.SAVED_MODEL_PATH))

        self.LOG_PATH = os.path.join(self.CURRENT_DIR, "logs")
        if not os.path.isdir(self.LOG_PATH):
            os.system("mkdir -p {}".format(self.LOG_PATH))
        _config_logging(log_file=os.path.join(self.LOG_PATH, "log.txt"))

    def display(self):
        """
        Display Configuration values.
        """
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


def _config_logging(log_file, log_level=logging.DEBUG):
    import sys

    format_line = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    custom_formatter = CustomFormatter(format_line)
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file)

    stream_handler.setFormatter(custom_formatter)

    logging.basicConfig(handlers=[file_handler, stream_handler], level=log_level, format=format_line)


class CustomFormatter(logging.Formatter):
    def format(self, record, *args, **kwargs):
        import copy

        LOG_COLORS = {
            logging.INFO: "\x1b[33m",
            logging.DEBUG: "\x1b[36m",
            logging.WARNING: "\x1b[31m",
            logging.ERROR: "\x1b[31;1m",
            logging.CRITICAL: "\x1b[35m",
        }

        new_record = copy.copy(record)
        if new_record.levelno in LOG_COLORS:
            new_record.levelname = "{color_begin}{level}{color_end}".format(
                level=new_record.levelname, color_begin=LOG_COLORS[new_record.levelno], color_end="\x1b[0m",
            )
        return super(CustomFormatter, self).format(new_record, *args, **kwargs)
