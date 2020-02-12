#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch
from tensorboardX import SummaryWriter
from albumentations import *

from models import YoloNet
from config import Config
from trainer import Trainer
from data_loader import DataTransformBase


def train_process(args, total_config, dataset_class, data_transform_class, params):

    # --------------------------------------------------------------------------#
    # prepare dataset
    # --------------------------------------------------------------------------#

    def _worker_init_fn_():
        import random
        import numpy as np
        import torch

        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.random_seed)

    input_size = (params["img_h"], params["img_w"])

    transforms = [
        OneOf([IAAAdditiveGaussianNoise(), GaussNoise()], p=0.5),
        OneOf([MedianBlur(blur_limit=3), GaussianBlur(blur_limit=3), MotionBlur(blur_limit=3),], p=0.1,),
        RandomGamma(gamma_limit=(80, 120), p=0.5),
        RandomBrightnessContrast(p=0.5),
        HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        ChannelShuffle(p=0.5),
        HorizontalFlip(p=0.5),
        Cutout(num_holes=5, max_w_size=40, max_h_size=40, p=0.5),
        Rotate(limit=20, p=0.5, border_mode=0),
    ]

    data_transform = data_transform_class(transforms=transforms, input_size=input_size)
    train_dataset = dataset_class(
        data_path=total_config.DATA_PATH,
        phase="train",
        normalize_bbox=True,
        transform=[data_transform],
        multiscale=args.multiscale,
        resize_after_batch_num=args.resize_after_batch_num,
    )

    val_dataset = dataset_class(
        data_path=total_config.DATA_PATH, phase="val", normalize_bbox=True, transform=data_transform,
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.od_collate_fn,
        num_workers=args.num_workers,
        drop_last=True,
        worker_init_fn=_worker_init_fn_(),
    )
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=val_dataset.od_collate_fn,
        num_workers=args.num_workers,
        drop_last=True,
    )
    data_loaders_dict = {"train": train_data_loader, "val": val_data_loader}

    # --------------------------------------------------------------------------#
    # configuration for training
    # --------------------------------------------------------------------------#

    tblogger = SummaryWriter(total_config.LOG_PATH)

    model = YoloNet(dataset_config=params)
    if args.backbone_weight_path:
        model.feature.load_weight(args.backbone_weight_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = None

    base_lr_rate = args.lr_rate / (args.batch_size * args.batch_multiplier)
    base_weight_decay = args.weight_decay * (args.batch_size * args.batch_multiplier)
    steps = [float(v.strip()) for v in args.steps.split(",")]
    scales = [float(v.strip()) for v in args.scales.split(",")]

    def adjust_learning_rate(optimizer, processed_batch):
        lr = base_lr_rate
        for i in range(len(steps)):
            scale = scales[i] if i < len(scales) else 1
            if processed_batch >= steps[i]:
                lr = lr * scale
                if processed_batch == steps[i]:
                    break
                else:
                    break
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr / args.batch_size
        return lr

    optimizer = torch.optim.SGD(
        model.parameters(), lr=base_lr_rate, momentum=args.momentum, weight_decay=base_weight_decay,
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        metric_func=None,
        optimizer=optimizer,
        num_epochs=args.num_epoch,
        save_period=args.save_period,
        config=total_config,
        data_loaders_dict=data_loaders_dict,
        device=device,
        dataset_name_base=train_dataset.__name__,
        batch_multiplier=args.batch_multiplier,
        adjust_lr_callback=adjust_learning_rate,
        logger=tblogger,
    )

    if args.snapshot and os.path.isfile(args.snapshot):
        trainer.resume_checkpoint(args.snapshot)

    with torch.autograd.set_detect_anomaly(True):
        trainer.train()

    tblogger.close()


def main(args):
    total_config = Config()
    total_config.display()
    if not args.dataset or args.dataset not in total_config.DATASETS.keys():
        raise Exception("specify one of the datasets to use in {}".format(list(total_config.DATASETS.keys())))

    dataset = args.dataset
    dataset_class = total_config.DATASETS[dataset]
    data_transform_class = DataTransformBase
    params = total_config.DATASET_PARAMS[dataset]
    train_process(args, total_config, dataset_class, data_transform_class, params)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epoch", type=int, default=300)
    parser.add_argument("--lr_rate", type=float, default=1e-3)
    parser.add_argument("--batch_multiplier", type=int, default=1)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=5e-4, type=float)
    parser.add_argument("--burn_in", default=1000, type=int)
    parser.add_argument("--steps", default="40000,45000", type=str)
    parser.add_argument("--scales", default=".1,.1", type=str)
    parser.add_argument("--gamma", default=0.1, type=float)
    parser.add_argument("--milestones", default="120, 220", type=str)
    parser.add_argument("--save_period", type=int, default=1)
    parser.add_argument("--backbone_weight_path", type=str)
    parser.add_argument("--multiscale", type=bool, default=True)
    parser.add_argument("--resize_after_batch_num", type=int, default=10)
    parser.add_argument("--snapshot", type=str, help="path to snapshot weights")
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        help="name of the dataset to use",
        choices=["bird_dataset", "switch_dataset", "wheat_dataset"],
    )
    parser.add_argument("--random_seed", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=4)
    parsed_args = parser.parse_args()

    main(parsed_args)
