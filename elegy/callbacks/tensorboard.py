# Implementation based on tf.keras.callbacks.py
# https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/callbacks.py


import collections
import typing as tp
from typing import Optional, Union

import numpy as np
import six
from tensorboardX.writer import SummaryWriter

from .callback import Callback


class TensorBoard(Callback):
    """Callback that streams epoch results to tensorboard events folder.

Supports all values that can be represented as a string,
including 1D iterables such as `np.ndarray`.

Example:

    ```python
    tensorboard_logger = TensorBoard('runs')
    model.fit(X_train, Y_train, callbacks=[tensorboard_logger])
    ```
  """

    def __init__(
        self,
        logdir: Optional[str] = None,
        update_freq: Union[str, int] = "epoch",
        **kwargs
    ) -> None:
        """
        Arguments:
            logdir: Save directory location. Default is
                runs/**CURRENT_DATETIME_HOSTNAME**, which changes after each run.
                Use hierarchical folder structure to compare
                between runs easily. e.g. pass in 'runs/exp1', 'runs/exp2', etc.
                for each new experiment to compare across them.
            update_freq: `'batch'` or `'epoch'` or integer. When using `'batch'`,
                writes the losses and metrics to TensorBoard after each batch. The same
                applies for `'epoch'`. 
            **kwargs: Options to pass to `SummaryWriter` object
        """
        self.logdir = logdir
        self.writer = None
        self.keys = None
        if update_freq == "batch":
            self.update_freq = 1
        else:
            self.update_freq = update_freq
        self._open_args = kwargs if kwargs else {}
        super(TensorBoard, self).__init__()

    def on_train_begin(self, logs=None):
        self.writer = SummaryWriter(self.logdir, **self._open_args)

    def on_train_batch_end(self, batch: int, logs):
        if self.update_freq == "epoch":
            return
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, tp.Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (", ".join(map(str, k)))
            else:
                return k

        if self.update_freq != "epoch" and batch % self.update_freq == 0:
            if self.keys is None:
                self.keys = sorted(logs.keys())
            row_dict = collections.OrderedDict({"batch": batch})
            row_dict.update(
                (key + "batch", handle_value(logs[key])) for key in self.keys
            )

            for key, value in row_dict.items():
                self.writer.add_scalar(key, value, batch)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, tp.Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (", ".join(map(str, k)))
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())

        row_dict = collections.OrderedDict({"epoch": epoch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)

        for key, value in row_dict.items():
            self.writer.add_scalar(key, value, epoch)

    def on_train_end(self, logs=None):
        self.writer.close()
