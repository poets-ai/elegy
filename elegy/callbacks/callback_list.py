# Implementation based on tf.keras.callbacks.py
# https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/callbacks.py

import collections
import logging
import time
import typing as tp

import numpy as np

from .callback import Callback
from .history import History
from .progbar_logger import ProgbarLogger


class ModeKeys(object):
    """Standard names for model modes.

    The following standard keys are defined:

    * `TRAIN`: training/fitting mode.
    * `TEST`: testing/evaluation mode.
    * `PREDICT`: prediction/inference mode.
    """

    TRAIN = "train"
    TEST = "test"
    PREDICT = "predict"


class CallbackList(object):
    """Container abstracting a list of callbacks."""

    def __init__(
        self,
        callbacks: tp.Optional[tp.List[Callback]] = None,
        add_history: bool = False,
        add_progbar: bool = False,
        model: tp.Optional[tp.Any] = None,
        **params
    ):
        """Creates a container for `Callbacks`.

        Arguments:
          callbacks: List of `Callback` instances.
          add_history: Whether a `History` callback should be added, if one does not
            already exist in `callback`s.
          add_progbar: Whether a `ProgbarLogger` callback should be added, if one
            does not already exist in `callback`s.
          model: The `Model` these `Callback`s are used with.`
          **params: If provided, parameters will be passed to each `Callback` via
            `Callback.set_params`.
        """
        self.callbacks = callbacks if callbacks else []
        self._add_default_callbacks(add_history, add_progbar)

        if model:
            self.set_model(model)
        if params:
            self.set_params(params)

        self._queue_length = 10
        self._reset_batch_timing()

        # Determines if batch-level hooks need to be called.
        # This is important for performance, because processing batch-level logs
        # will cause async eager to block on each batch.
        # pylint: disable=protected-access
        self._should_call_train_batch_hooks = any(
            cb._implements_train_batch_hooks() for cb in self.callbacks
        )
        self._should_call_test_batch_hooks = any(
            cb._implements_test_batch_hooks() for cb in self.callbacks
        )
        self._should_call_predict_batch_hooks = any(
            cb._implements_predict_batch_hooks() for cb in self.callbacks
        )
        # pylint: enable=protected-access

    def _add_default_callbacks(self, add_history, add_progbar):
        """Adds `Callback`s that are always present."""
        self._progbar = None
        self._history = None

        for cb in self.callbacks:
            if isinstance(cb, ProgbarLogger):
                self._progbar = cb
            elif isinstance(cb, History):
                self._history = cb

        if self._progbar is None and add_progbar:
            self._progbar = ProgbarLogger(count_mode="steps")
            self.callbacks.append(self._progbar)

        if self._history is None and add_history:
            self._history = History()
            self.callbacks.append(self._history)

    def _reset_batch_timing(self):
        self._delta_t_batch = 0.0
        self._delta_ts = collections.defaultdict(
            lambda: collections.deque([], maxlen=self._queue_length)
        )

    def _process_logs(self, logs):
        """Turns tensors into numpy arrays or Python scalars."""
        if logs:
            return logs
        return {}

    def append(self, callback):
        self.callbacks.append(callback)

    def set_params(self, params):
        self.params = params
        for callback in self.callbacks:
            callback.set_params(params)

    def set_model(self, model):
        self.model = model
        if self._history:
            model.history = self._history
        for callback in self.callbacks:
            callback.set_model(model)

    def _call_batch_hook(self, mode, hook, batch, logs=None):
        """Helper function for all batch_{begin | end} methods."""
        if not self.callbacks:
            return
        hook_name = "on_{mode}_batch_{hook}".format(mode=mode, hook=hook)
        if hook == "begin":
            self._t_enter_batch = time.time()
        if hook == "end":
            # Batch is ending, calculate batch time.
            self._delta_t_batch = time.time() - self._t_enter_batch

        logs = logs or {}
        t_before_callbacks = time.time()
        for callback in self.callbacks:
            batch_hook = getattr(callback, hook_name)
            batch_hook(batch, logs)
        self._delta_ts[hook_name].append(time.time() - t_before_callbacks)

        # delta_t_median = np.median(self._delta_ts[hook_name])
        # if (
        #     self._delta_t_batch > 0.0
        #     and delta_t_median > 0.95 * self._delta_t_batch
        #     and delta_t_median > 0.1
        # ):
        #     logging.warning(
        #         "Method (%s) is slow compared "
        #         "to the batch update (%f). Check your callbacks.",
        #         hook_name,
        #         delta_t_median,
        #     )

    def _call_begin_hook(self, mode):
        """Helper function for on_{train|test|predict}_begin methods."""
        if mode == ModeKeys.TRAIN:
            self.on_train_begin()
        elif mode == ModeKeys.TEST:
            self.on_test_begin()
        else:
            self.on_predict_begin()

    def _call_end_hook(self, mode):
        """Helper function for on_{train|test|predict}_end methods."""
        if mode == ModeKeys.TRAIN:
            self.on_train_end()
        elif mode == ModeKeys.TEST:
            self.on_test_end()
        else:
            self.on_predict_end()

    def on_batch_begin(self, batch, logs=None):
        if self._should_call_train_batch_hooks:
            logs = self._process_logs(logs)
            self._call_batch_hook(ModeKeys.TRAIN, "begin", batch, logs=logs)

    def on_batch_end(self, batch, logs=None):
        if self._should_call_train_batch_hooks:
            logs = self._process_logs(logs)
            self._call_batch_hook(ModeKeys.TRAIN, "end", batch, logs=logs)

    def on_epoch_begin(self, epoch, logs=None):
        """Calls the `on_epoch_begin` methods of its callbacks.

        This function should only be called during TRAIN mode.

        Arguments:
            epoch: integer, index of epoch.
            logs: dict. Currently no data is passed to this argument for this method
                but that may change in the future.
        """
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
        self._reset_batch_timing()

    def on_epoch_end(self, epoch, logs=None):
        """Calls the `on_epoch_end` methods of its callbacks.

        This function should only be called during TRAIN mode.

        Arguments:
            epoch: integer, index of epoch.
            logs: dict, metric results for this training epoch, and for the
                validation epoch if validation is performed. Validation result keys
                are prefixed with `val_`.
        """
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_train_batch_begin(self, batch, logs=None):
        """Calls the `on_train_batch_begin` methods of its callbacks.

        Arguments:
            batch: integer, index of batch within the current epoch.
            logs: dict. Has keys `batch` and `size` representing the current batch
                number and the size of the batch.
        """
        # TODO(b/150629188): Make ProgBarLogger callback not use batch hooks
        # when verbose != 1
        if self._should_call_train_batch_hooks:
            logs = self._process_logs(logs)
            self._call_batch_hook(ModeKeys.TRAIN, "begin", batch, logs=logs)

    def on_train_batch_end(self, batch, logs=None):
        """Calls the `on_train_batch_end` methods of its callbacks.

        Arguments:
            batch: integer, index of batch within the current epoch.
            logs: dict. Metric results for this batch.
        """
        if self._should_call_train_batch_hooks:
            logs = self._process_logs(logs)
            self._call_batch_hook(ModeKeys.TRAIN, "end", batch, logs=logs)

    def on_test_batch_begin(self, batch, logs=None):
        """Calls the `on_test_batch_begin` methods of its callbacks.

        Arguments:
            batch: integer, index of batch within the current epoch.
            logs: dict. Has keys `batch` and `size` representing the current batch
                number and the size of the batch.
        """
        if self._should_call_test_batch_hooks:
            logs = self._process_logs(logs)
            self._call_batch_hook(ModeKeys.TEST, "begin", batch, logs=logs)

    def on_test_batch_end(self, batch, logs=None):
        """Calls the `on_test_batch_end` methods of its callbacks.

        Arguments:
            batch: integer, index of batch within the current epoch.
            logs: dict. Metric results for this batch.
        """
        if self._should_call_test_batch_hooks:
            logs = self._process_logs(logs)
            self._call_batch_hook(ModeKeys.TEST, "end", batch, logs=logs)

    def on_predict_batch_begin(self, batch, logs=None):
        """Calls the `on_predict_batch_begin` methods of its callbacks.

        Arguments:
            batch: integer, index of batch within the current epoch.
            logs: dict. Has keys `batch` and `size` representing the current batch
                number and the size of the batch.
        """
        if self._should_call_predict_batch_hooks:
            logs = self._process_logs(logs)
            self._call_batch_hook(ModeKeys.PREDICT, "begin", batch, logs=logs)

    def on_predict_batch_end(self, batch, logs=None):
        """Calls the `on_predict_batch_end` methods of its callbacks.

        Arguments:
            batch: integer, index of batch within the current epoch.
            logs: dict. Metric results for this batch.
        """
        if self._should_call_predict_batch_hooks:
            logs = self._process_logs(logs)
            self._call_batch_hook(ModeKeys.PREDICT, "end", batch, logs=logs)

    def on_train_begin(self, logs=None):
        """Calls the `on_train_begin` methods of its callbacks.

        Arguments:
            logs: dict. Currently no data is passed to this argument for this method
                but that may change in the future.
        """
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        """Calls the `on_train_end` methods of its callbacks.

        Arguments:
            logs: dict. Currently no data is passed to this argument for this method
                but that may change in the future.
        """
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_test_begin(self, logs=None):
        """Calls the `on_test_begin` methods of its callbacks.

        Arguments:
            logs: dict. Currently no data is passed to this argument for this method
                but that may change in the future.
        """
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_test_begin(logs)

    def on_test_end(self, logs=None):
        """Calls the `on_test_end` methods of its callbacks.

        Arguments:
            logs: dict. Currently no data is passed to this argument for this method
                but that may change in the future.
        """
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_test_end(logs)

    def on_predict_begin(self, logs=None):
        """Calls the 'on_predict_begin` methods of its callbacks.

        Arguments:
            logs: dict. Currently no data is passed to this argument for this method
                but that may change in the future.
        """
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_predict_begin(logs)

    def on_predict_end(self, logs=None):
        """Calls the `on_predict_end` methods of its callbacks.

        Arguments:
            logs: dict. Currently no data is passed to this argument for this method
                but that may change in the future.
        """
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_predict_end(logs)

    def __iter__(self):
        return iter(self.callbacks)
