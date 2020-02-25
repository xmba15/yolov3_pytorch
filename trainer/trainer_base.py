#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
import sys
import torch
from abc import abstractmethod

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_DIR, ".."))


class TrainerBase:
    def __init__(
        self,
        model,
        criterion,
        metric_func,
        optimizer,
        num_epochs,
        save_period,
        config,
        device=None,
        dataset_name_base="",
        batch_multiplier=1,
        logger=None,
    ):
        self._model = model
        self.criterion = criterion
        self._metric_func = metric_func
        self._optimizer = optimizer
        self._config = config
        self._dataset_name_base = dataset_name_base
        self._batch_multiplier = batch_multiplier
        self._logger = logger

        if device is None:
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self._device = device

        self._model = self._model.to(self._device)

        self._config = config

        self._checkpoint_dir = self._config.SAVED_MODEL_PATH

        self._start_epoch = 1

        self._num_epochs = num_epochs

        self._save_period = save_period

    @property
    def model(self):
        return self._model

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _save_checkpoint(self, epoch, save_best=False):
        arch = type(self._model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
        }
        output_file = "checkpoint_{}_epoch_{}.pth".format(arch, epoch)
        if self._dataset_name_base and isinstance(self._dataset_name_base, str) and self._dataset_name_base != "":
            output_file = "{}_{}".format(self._dataset_name_base, output_file)

        filename = os.path.join(self._checkpoint_dir, output_file)
        torch.save(state, filename)

        # if save_best:
        #     best_path = os.path.join(self._checkpoint_dir, "model_best.pth")
        #     torch.save(state, best_path)

    def resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)

        checkpoint = torch.load(resume_path)
        self._start_epoch = checkpoint["epoch"] + 1

        self._model.load_state_dict(checkpoint["state_dict"])

        self._optimizer.load_state_dict(checkpoint["optimizer"])

    def train(self):
        logging.info("========================================")
        logging.info("Start training {}".format(type(self._model).__name__))
        logging.info("========================================")
        logs = []

        for epoch in range(self._start_epoch, self._num_epochs + 1):
            train_loss, val_loss = self._train_epoch(epoch)

            log_epoch = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }

            logging.info(
                "\n----------------------------------------------------\n"
                + "epoch: {}, train_loss: {: .4f}, val_loss: {: .4f}".format(epoch, train_loss, val_loss)
                + "\n----------------------------------------------------\n"
            )
            logs.append(log_epoch)
            if self._logger:
                self._logger.add_scalar("train/train_loss", train_loss, epoch)
                self._logger.add_scalar("val/val_loss", val_loss, epoch)

            if (epoch + 1) % self._save_period == 0:
                self._save_checkpoint(epoch, save_best=True)

        return logs
