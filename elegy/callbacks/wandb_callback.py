# Implementation based on tf.keras.callbacks.py and elegy.callbacks.TensorBoard
# https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/callbacks.py
# https://github.com/poets-ai/elegy/blob/master/elegy/callbacks/tensorboard.py


from typing import Union
from wandb.sdk import wandb_run

from .callback import Callback


class WandbCallback(Callback):
    """
    Callback that streams epoch results to a [Weights & Biases](https://wandb.ai/) run.

    ```python
    run = wandb.init(project="sample-wandb-project")
    wandb_logger = WandbCallback(run=run)
    model.fit(X_train, Y_train, callbacks=[wandb_logger])
    ```
    """

    def __init__(
        self, run: wandb_run.Run, update_freq: Union[str, int] = "epoch"
    ):
        """
        Arguments:
            run: Weights and Biases Run of type `wandb.sdk.wandb_run.Run`. The Run
                object can be initialized by invoking `wandb.init()`.
            update_freq: `'batch'` or `'epoch'` or integer. When using `'batch'`,
                writes the losses and metrics to TensorBoard after each batch. The same
                applies for `'epoch'`. If using an integer, let's say `1000`, the
                callback will write the metrics and losses to TensorBoard every 1000
                batches. Note that writing too frequently to TensorBoard can slow down
                your training.
        """
        super().__init__()
        self.run = run
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

    def on_train_begin(self, logs=None):
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
                self.run.log({key: logs[key]}, step=self.global_step)

    def on_epoch_begin(self, epoch: int, logs=None):
        self.current_epoch = epoch

    def on_epoch_end(self, epoch: int, logs=None):
        logs = logs or {}
        if self.keys is None:
            self.keys = logs.keys()

        if self.write_per_batch:
            for key in logs:
                self.run.log({key: logs[key]}, step=self.global_step)
            return

        elif epoch % self.update_freq == 0:
            for key in logs:
                self.run.log({key: logs[key]}, step=epoch)
