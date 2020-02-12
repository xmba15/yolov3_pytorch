#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import torch
import torch.nn as nn
import logging
from .trainer_base import TrainerBase


_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_DIR, ".."))
try:
    from utils import inf_loop
except Exception as e:
    print(e)
    sys.exit(-1)


class Trainer(TrainerBase):
    def __init__(
        self,
        model,
        criterion,
        metric_func,
        optimizer,
        num_epochs,
        save_period,
        config,
        data_loaders_dict,
        scheduler=None,
        device=None,
        len_epoch=None,
        dataset_name_base="",
        batch_multiplier=1,
        logger=None,
        processed_batch=0,
        adjust_lr_callback=None,
        print_after_batch_num=10,
    ):
        super(Trainer, self).__init__(
            model,
            criterion,
            metric_func,
            optimizer,
            num_epochs,
            save_period,
            config,
            device,
            dataset_name_base,
            batch_multiplier,
            logger,
        )

        self.train_data_loader = data_loaders_dict["train"]
        self.val_data_loader = data_loaders_dict["val"]

        self.num_train_imgs = len(self.train_data_loader.dataset)
        self.num_val_imgs = len(self.val_data_loader.dataset)

        self.processed_batch = processed_batch
        self.adjust_lr_callback = adjust_lr_callback

        if len_epoch is None:
            self._len_epoch = len(self.train_data_loader)
        else:
            self.train_data_loader = inf_loop(self.train_data_loader)
            self._len_epoch = len_epoch

        self._do_validation = self.val_data_loader is not None
        self._scheduler = scheduler

        self._print_after_batch_num = print_after_batch_num

        self._stat_keys = ["loss", "loss_x", "loss_y", "loss_w", "loss_h", "loss_conf", "loss_cls"]

    def _train_epoch(self, epoch):
        self._model.train()

        batch_size = self.train_data_loader.batch_size

        epoch_train_loss = 0.0
        count = self._batch_multiplier
        running_losses = dict(zip(self._stat_keys, [0.0] * len(self._stat_keys)))

        for batch_idx, (data, target, length_tensor) in enumerate(self.train_data_loader):

            if self.adjust_lr_callback is not None:
                self.adjust_lr_callback(self._optimizer, self.processed_batch)
                self.processed_batch += 1

            data = data.to(self._device)
            target = target.to(self._device)
            length_tensor = length_tensor.to(self._device)

            if count == 0:
                self._optimizer.step()
                self._optimizer.zero_grad()
                count = self._batch_multiplier

            with torch.set_grad_enabled(True):
                output = self._model(data)

                train_loss = self._model(data, target)
                for key in self._stat_keys:
                    running_losses[key] += self._model.stats[key]

                total_loss = train_loss / self._batch_multiplier
                total_loss.backward()
                count -= 1

                if (batch_idx + 1) % self._print_after_batch_num == 0:
                    logging.info(
                        "\n epoch: {}/{} || iter: {}/{} || [Losses: total: {}, loss_x: {}, loss_y: {}, loss_w: {}, loss_h: {}, loss_conf: {}, loss_cls: {} || lr_rate: {}".format(
                            epoch,
                            self._num_epochs,
                            batch_idx,
                            len(self.train_data_loader),
                            running_losses["loss"] / self._print_after_batch_num,
                            running_losses["loss_x"] / self._print_after_batch_num,
                            running_losses["loss_y"] / self._print_after_batch_num,
                            running_losses["loss_w"] / self._print_after_batch_num,
                            running_losses["loss_h"] / self._print_after_batch_num,
                            running_losses["loss_conf"] / self._print_after_batch_num,
                            running_losses["loss_cls"] / self._print_after_batch_num,
                            self._optimizer.param_groups[0]["lr"],
                        )
                    )
                    running_losses = dict(zip(self._stat_keys, [0.0] * len(self._stat_keys)))

                epoch_train_loss += total_loss.item() * self._batch_multiplier

            if batch_idx == self._len_epoch:
                break

        if self._do_validation:
            epoch_val_loss = self._valid_epoch(epoch)

        if self._scheduler is not None:
            self._scheduler.step()

        return (
            epoch_train_loss / self.num_train_imgs,
            epoch_val_loss / self.num_val_imgs,
        )

    def _valid_epoch(self, epoch):
        print("start validation...")
        self._model.eval()

        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (data, target, length_tensor) in enumerate(self.val_data_loader):

                data = data.to(self._device)
                target = target.to(self._device)
                length_tensor = length_tensor.to(self._device)

                val_loss = self._model(data, target)
                epoch_val_loss += val_loss.item()

        return epoch_val_loss
