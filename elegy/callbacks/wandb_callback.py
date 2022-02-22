# Implementation based on tf.keras.callbacks.py and elegy.callbacks.TensorBoard
# https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/callbacks.py
# https://github.com/poets-ai/elegy/blob/master/elegy/callbacks/tensorboard.py


import wandb
from typing import Union, Optional, Dict

from .callback import Callback


class WandbCallback(Callback):
    """
    Callback that streams epoch results to a [Weights & Biases](https://wandb.ai/) run.

    ```python
    wandb.login()
    wandb_logger = WandbCallback(
        project="sample-wandb-project",
        entity="myself",
        job_type="train"
    )
    model.fit(X_train, Y_train, callbacks=[wandb_logger])
    ```
    """

    def __init__(
        self,
        project: Optional[str] = None,
        name: Optional[str] = None,
        entity: Optional[str] = None,
        job_type: Optional[str] = None,
        config: Union[Dict, str, None] = None,
        update_freq: Union[str, int] = "epoch",
        **kwargs
    ):
        """
        Arguments:
            project: (str, optional) The name of the project where you're sending the new run.
                If the project is not specified, the run is put in an "Uncategorized" project.
            name: (str, optional) A short display name for this run, which is how you'll
                identify this run in the UI. By default we generate a random two-word name that
                lets you easily cross-reference runs from the table to charts. Keeping these run
                names short makes the chart legends and tables easier to read. If you're looking
                for a place to save your hyperparameters, we recommend saving those in config.
            entity: (str, optional) An entity is a username or team name where you're sending runs.
                This entity must exist before you can send runs there, so make sure to create your
                account or team in the UI before starting to log runs. If you don't specify an entity,
                the run will be sent to your default entity, which is usually your username.
            job_type: (str, optional) Specify the type of run, which is useful when you're grouping
                runs together into larger experiments using group. For example, you might have multiple
                jobs in a group, with job types like train and eval. Setting this makes it easy to
                filter and group similar runs together in the UI so you can compare apples to apples.
            config: (dict, argparse, absl.flags, str, optional) This sets `wandb.config`, a dictionary-like
                object for saving inputs to your job, like hyperparameters for a model or settings for a
                data preprocessing job. The config will show up in a table in the UI that you can use to
                group, filter, and sort runs. Keys should not contain . in their names, and values should
                be under 10 MB. If dict, argparse or `absl.flags`: will load the key value pairs into the
                wandb.config object. If str: will look for a yaml file by that name, and load config from
                that file into the `wandb.config` object.
            update_freq: (str, int)`'batch'` or `'epoch'` or integer. When using `'batch'`, writes the
                losses and metrics to TensorBoard after each batch. The same applies for `'epoch'`. If
                using an integer, let's say `1000`, the callback will write the metrics and losses to
                TensorBoard every 1000 batches. Note that writing too frequently to TensorBoard can slow
                down your training.
        """
        super().__init__()
        self.run = wandb.init(
            project=project,
            name=name,
            entity=entity,
            job_type=job_type,
            config=config,
            **kwargs
        ) if wandb.run is None else wandb.run
        self.keys = None
        self.write_per_batch = True
        self._ignore_fields = ["size"]
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
                if key in self._ignore_fields or "val_" + key in self._ignore_fields:
                    continue
                log_key = "train_" + key if key[:4] != "val_" else key
                self.run.log({log_key: logs[key]}, step=self.global_step)

    def on_epoch_begin(self, epoch: int, logs=None):
        self.current_epoch = epoch

    def on_epoch_end(self, epoch: int, logs=None):
        logs = logs or {}
        if self.keys is None:
            self.keys = logs.keys()

        if self.write_per_batch:
            for key in logs:
                if key in self._ignore_fields or "val_" + key in self._ignore_fields:
                    continue
                log_key = "train_" + key if key[:4] != "val_" else key
                self.run.log({log_key: logs[key]}, step=self.global_step)
            return

        elif epoch % self.update_freq == 0:
            for key in logs:
                if key in self._ignore_fields or "val_" + key in self._ignore_fields:
                    continue
                log_key = "train_" + key if key[:4] != "val_" else key
                self.run.log({log_key: logs[key]}, step=epoch)
    
    def on_train_end(self, logs=None):
        self.run.finish()
