# Implementation based on tf.keras.callbacks.py
# https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/callbacks.py

import typing as tp

import numpy as np


def default(method):
    """Decorates a method to detect overrides in subclasses."""
    method._is_default = True  # pylint: disable=protected-access
    return method


def is_default(method):
    """Check if a method is decorated with the `default` wrapper."""
    return getattr(method, "_is_default", False)


class Callback(object):
    """Abstract base class used to build new callbacks.

    The `logs` dictionary that callback methods
    take as argument will contain keys for quantities relevant to
    the current batch or epoch.

    Currently, the `.fit()` method of the `Model` class
    will include the following quantities in the `logs` that
    it passes to its callbacks:

    ```python
    on_epoch_end: logs include `acc` and `loss`, and
        optionally include `val_loss`
        (if validation is enabled in `fit`), and `val_acc`
        (if validation and accuracy monitoring are enabled).
    on_train_batch_begin: logs include `size`,
        the number of samples in the current batch.
    on_train_batch_end: logs include `loss`, and optionally `acc`
        (if accuracy monitoring is enabled).
    ```

    Attributes:
        params (dict):  Training parameters
            (eg. verbosity, batch size, number of epochs...).
        model (elegy.model.Model): Reference of the model being trained.
    """

    __all__ = [
        "on_epoch_begin",
        "on_epoch_end",
        "on_predict_batch_begin",
        "on_predict_batch_end",
        "on_predict_begin",
        "on_predict_end",
        "on_test_batch_begin",
        "on_test_batch_end",
        "on_test_begin",
        "on_test_end",
        "on_train_batch_begin",
        "on_train_batch_end",
        "on_train_begin",
        "on_train_end",
        "set_model",
        "set_params",
    ]

    def __init__(self):
        self.model = None
        # Whether this Callback should only run on the chief worker in a
        # Multi-Worker setting.
        self._chief_worker_only = None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    # @doc_controls.for_subclass_implementers
    def on_epoch_begin(
        self, epoch: int, logs: tp.Optional[tp.Dict[str, np.ndarray]] = None
    ):
        """Called at the start of an epoch.

        Subclasses should override for any actions to run. This function should only
        be called during TRAIN mode.

        Arguments:
            epoch: integer, index of epoch.
            logs: dict. Currently no data is passed to this argument for this method but
                that may change in the future.
        """
        pass

    # @doc_controls.for_subclass_implementers
    def on_epoch_end(
        self, epoch: int, logs: tp.Optional[tp.Dict[str, np.ndarray]] = None
    ):
        """Called at the end of an epoch.

        Subclasses should override for any actions to run. This function should only
        be called during TRAIN mode.

        Arguments:
            epoch: integer, index of epoch.
            logs: dict, metric results for this training epoch, and for the
                validation epoch if validation is performed. Validation result keys
                are prefixed with `val_`.
        """
        pass

    # @doc_controls.for_subclass_implementers
    @default
    def on_train_batch_begin(
        self, batch: int, logs: tp.Optional[tp.Dict[str, np.ndarray]] = None
    ):
        """Called at the beginning of a training batch in `fit` methods.

        Subclasses should override for any actions to run.

        Arguments:
            batch: integer, index of batch within the current epoch.
            logs: dict. Has keys `batch` and `size` representing the current batch
                number and the size of the batch.
        """
        pass

    # @doc_controls.for_subclass_implementers
    @default
    def on_train_batch_end(
        self, batch: int, logs: tp.Optional[tp.Dict[str, np.ndarray]] = None
    ):
        """Called at the end of a training batch in `fit` methods.

        Subclasses should override for any actions to run.

        Arguments:
            batch: integer, index of batch within the current epoch.
            logs: dict. Metric results for this batch.
        """
        pass

    # @doc_controls.for_subclass_implementers
    @default
    def on_test_batch_begin(
        self, batch: int, logs: tp.Optional[tp.Dict[str, np.ndarray]] = None
    ):
        """Called at the beginning of a batch in `evaluate` methods.

        Also called at the beginning of a validation batch in the `fit`
        methods, if validation data is provided.

        Subclasses should override for any actions to run.

        Arguments:
            batch: integer, index of batch within the current epoch.
            logs: dict. Has keys `batch` and `size` representing the current batch
                number and the size of the batch.
        """
        pass

    # @doc_controls.for_subclass_implementers
    @default
    def on_test_batch_end(
        self, batch: int, logs: tp.Optional[tp.Dict[str, np.ndarray]] = None
    ):
        """Called at the end of a batch in `evaluate` methods.

        Also called at the end of a validation batch in the `fit`
        methods, if validation data is provided.

        Subclasses should override for any actions to run.

        Arguments:
            batch: integer, index of batch within the current epoch.
            logs: dict. Metric results for this batch.
        """
        pass

    # @doc_controls.for_subclass_implementers
    @default
    def on_predict_batch_begin(
        self, batch: int, logs: tp.Optional[tp.Dict[str, np.ndarray]] = None
    ):
        """Called at the beginning of a batch in `predict` methods.

        Subclasses should override for any actions to run.

        Arguments:
            batch: integer, index of batch within the current epoch.
            logs: dict. Has keys `batch` and `size` representing the current batch
                number and the size of the batch.
        """
        pass

    # @doc_controls.for_subclass_implementers
    @default
    def on_predict_batch_end(
        self, batch: int, logs: tp.Optional[tp.Dict[str, np.ndarray]] = None
    ):
        """Called at the end of a batch in `predict` methods.

        Subclasses should override for any actions to run.

        Arguments:
            batch: integer, index of batch within the current epoch.
            logs: dict. Metric results for this batch.
        """
        pass

    # @doc_controls.for_subclass_implementers
    def on_train_begin(self, logs: tp.Optional[tp.Dict[str, np.ndarray]] = None):
        """Called at the beginning of training.

        Subclasses should override for any actions to run.

        Arguments:
            logs: dict. Currently no data is passed to this argument for this method
                but that may change in the future.
        """
        pass

    # @doc_controls.for_subclass_implementers
    def on_train_end(self, logs: tp.Optional[tp.Dict[str, np.ndarray]] = None):
        """Called at the end of training.

        Subclasses should override for any actions to run.

        Arguments:
            logs: dict. Currently no data is passed to this argument for this method
                but that may change in the future.
        """
        pass

    # @doc_controls.for_subclass_implementers
    def on_test_begin(self, logs: tp.Optional[tp.Dict[str, np.ndarray]] = None):
        """Called at the beginning of evaluation or validation.

        Subclasses should override for any actions to run.

        Arguments:
            logs: dict. Currently no data is passed to this argument for this method
                but that may change in the future.
        """
        pass

    # @doc_controls.for_subclass_implementers
    def on_test_end(self, logs: tp.Optional[tp.Dict[str, np.ndarray]] = None):
        """Called at the end of evaluation or validation.

        Subclasses should override for any actions to run.

        Arguments:
            logs: dict. Currently no data is passed to this argument for this method
                but that may change in the future.
        """
        pass

    # @doc_controls.for_subclass_implementers
    def on_predict_begin(self, logs: tp.Optional[tp.Dict[str, np.ndarray]] = None):
        """Called at the beginning of prediction.

        Subclasses should override for any actions to run.

        Arguments:
            logs: dict. Currently no data is passed to this argument for this method
                but that may change in the future.
        """
        pass

    # @doc_controls.for_subclass_implementers
    def on_predict_end(self, logs: tp.Optional[tp.Dict[str, np.ndarray]] = None):
        """Called at the end of prediction.

        Subclasses should override for any actions to run.

        Arguments:
            logs: dict. Currently no data is passed to this argument for this method
                but that may change in the future.
        """
        pass

    def _implements_train_batch_hooks(self):
        """Determines if this Callback should be called for each train batch."""
        return not is_default(self.on_train_batch_begin) or not is_default(
            self.on_train_batch_end
        )

    def _implements_test_batch_hooks(self):
        """Determines if this Callback should be called for each test batch."""
        return not is_default(self.on_test_batch_begin) or not is_default(
            self.on_test_batch_end
        )

    def _implements_predict_batch_hooks(self):
        """Determines if this Callback should be called for each predict batch."""
        return not is_default(self.on_predict_batch_begin) or not is_default(
            self.on_predict_batch_end
        )
