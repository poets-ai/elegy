# Implementation based on tf.keras.callbacks.py
# https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/callbacks.py


import os
import typing as tp
from typing import Any, Dict, Optional, Union
from tensorboardX.writer import SummaryWriter

from .callback import Callback


class TensorBoard(Callback):
    """
    Callback that streams epoch results to tensorboard events folder.

    Supports all values that can be represented as a string,
    including 1D iterables such as `np.ndarray`.


    ```python
    tensorboard_logger = TensorBoard('runs')
    model.fit(X_train, Y_train, callbacks=[tensorboard_logger])
    ```
    """

    def __init__(
        self,
        logdir: Optional[str] = None,
        *,
        update_freq: Union[str, int] = "epoch",
        purge_step: Optional[int] = None,
        comment: str = "",
    ) -> None:
        """
        Arguments:
            logdir: Save directory location. Default is
                runs/**CURRENT_DATETIME_HOSTNAME**/{train, val}, which changes after each run.
                Use hierarchical folder structure to compare
                between runs easily. e.g. pass in 'runs/exp1', 'runs/exp2', etc.
                for each new experiment to compare across them.
            update_freq: `'batch'` or `'epoch'` or integer. When using `'batch'`,
                writes the losses and metrics to TensorBoard after each batch. The same
                applies for `'epoch'`. If using an integer, let's say `1000`, the
                callback will write the metrics and losses to TensorBoard every 1000
                batches. Note that writing too frequently to TensorBoard can slow down
                your training.
            purge_step (int):
                When logging crashes at step :math:`T+X` and restarts at step :math:`T`,
                any events whose global_step larger or equal to :math:`T` will be
                purged and hidden from TensorBoard.
                Note that crashed and resumed experiments should have the same ``logdir``.
            comment (string): Comment logdir suffix appended to the default
                ``logdir``. If ``logdir`` is assigned, this argument has no effect.
        """
        if not logdir:
            import socket
            from datetime import datetime

            current_time = datetime.now().strftime("%b%d_%H-%M-%S")
            self.logdir = os.path.join(
                "runs", current_time + "_" + socket.gethostname() + comment
            )
        else:
            self.logdir = logdir
        self.train_writer = None
        self.val_writer = None
        self.keys = None
        self.write_per_batch = True
        try:
            self.update_freq = int(update_freq)
        except ValueError as e:
            self.update_freq = 1
            if update_freq == "batch":
                self.write_per_batch = True
            elif update_freq == "epoch":
                self.write_per_batch = False
            else:
                raise e
        self.purge_step = purge_step

        super(TensorBoard, self).__init__()

    def on_train_begin(self, logs=None):
        self.train_writer = SummaryWriter(
            os.path.join(self.logdir, "train"), purge_step=self.purge_step
        )
        self.val_writer = SummaryWriter(
            os.path.join(self.logdir, "val"), purge_step=self.purge_step
        )
        self.steps = self.params["steps"]
        self.global_step = 0

    def on_train_batch_end(self, batch: int, logs=None):
        if not self.write_per_batch:
            return
        logs = logs or {}
        self.global_step = batch + self.current_epoch * (self.steps)
        if self.global_step % self.update_freq == 0:
            if self.keys is None:
                self.keys = logs.keys()
            for key in self.keys:
                self.train_writer.add_scalar(key, logs[key], self.global_step)

    def on_epoch_begin(self, epoch: int, logs=None):
        self.current_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if self.keys is None:
            self.keys = logs.keys()

        # logs on on_{train, test}_batch_end do not have val metrics
        if self.write_per_batch:
            for key in logs:
                if "val" in key:
                    self.val_writer.add_scalar(
                        key.replace("val_", ""), logs[key], self.global_step
                    )
            return

        elif epoch % self.update_freq == 0:

            for key in self.keys:
                if "val" in key:
                    self.val_writer.add_scalar(
                        key.replace("val_", ""), logs[key], epoch
                    )
                else:
                    self.train_writer.add_scalar(key, logs[key], epoch)

    def on_train_end(self, logs=None):
        self.train_writer.close()
        self.val_writer.close()
