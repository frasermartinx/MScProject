
import os
from pathlib import Path
import sys
from typing import List, Union, Literal

import torch
import wandb

from neuralop.training.training_state import save_training_state
from neuralop.utils import compute_rank, compute_stable_rank, compute_explained_variance

from neuralop.training.callbacks import Callback



class BasicLoggerCallback2(Callback):
    """
    Callback that implements simple logging functionality
    expected when passing verbose to a Trainer
    """

    def __init__(self, wandb_kwargs=None):
        super().__init__()
        if wandb_kwargs:
            wandb.init(**wandb_kwargs)
        self.loss = []
        self.val_loss = []

    def on_init_end(self, *args, **kwargs):
        self._update_state_dict(**kwargs)

    def on_train_start(self, **kwargs):
        self._update_state_dict(**kwargs)

        train_loader = self.state_dict["train_loader"]
        test_loaders = self.state_dict["test_loaders"]
        verbose = self.state_dict["verbose"]

        n_train = len(train_loader.dataset)
        self._update_state_dict(n_train=n_train)

        if not isinstance(test_loaders, dict):
            test_loaders = dict(test=test_loaders)

        if verbose:
            print(f"Training on {n_train} samples")
            print(
                f"Testing on {[len(loader.dataset) for loader in test_loaders.values()]} samples"
                f"         on resolutions {[name for name in test_loaders]}."
            )
            sys.stdout.flush()

    def on_epoch_start(self, epoch):
        self._update_state_dict(epoch=epoch)

    def on_batch_start(self, idx, **kwargs):
        self._update_state_dict(idx=idx)

    def on_before_loss(self, out, **kwargs):
        if (
            self.state_dict["epoch"] == 0
            and self.state_dict["idx"] == 0
            and self.state_dict["verbose"]
        ):
            print(f"Raw outputs of size {out.shape=}")

    def on_before_val(self, epoch, train_err, time, avg_loss, avg_lasso_loss, **kwargs):
        # track training err and val losses to print at interval epochs
        msg = f"[{epoch}] time={time:.2f}, avg_loss={avg_loss:.4f}, train_err={train_err:.4f}"
        values_to_log = dict(train_err=train_err, time=time, avg_loss=avg_loss)
        self._update_state_dict(msg=msg, values_to_log=values_to_log)
        self._update_state_dict(avg_lasso_loss=avg_lasso_loss)
        self.loss.append(avg_loss)

    def on_val_epoch_end(self, errors, **kwargs):
        for loss_name, loss_value in errors.items():
            if isinstance(loss_value, float):
                self.state_dict["msg"] += f", {loss_name}={loss_value:.4f}"
            else:
                loss_value = {i: e.item() for (i, e) in enumerate(loss_value)}
                self.state_dict["msg"] += f", {loss_name}={loss_value}"
            self.state_dict["values_to_log"][loss_name] = loss_value
            self.val_loss.append(loss_value)
        
            

    def on_val_end(self, *args, **kwargs):
        if self.state_dict.get("regularizer", False):
            avg_lasso = self.state_dict.get("avg_lasso_loss", 0.0)
            avg_lasso /= self.state_dict.get("n_epochs")
            self.state_dict["msg"] += f", avg_lasso={avg_lasso:.5f}"

        print(self.state_dict["msg"])
        sys.stdout.flush()

        if self.state_dict.get("wandb_log", False):
            for pg in self.state_dict["optimizer"].param_groups:
                lr = pg["lr"]
                self.state_dict["values_to_log"]["lr"] = lr
            wandb.log(
                self.state_dict["values_to_log"],
                step=self.state_dict["epoch"] + 1,
                commit=True,
            )