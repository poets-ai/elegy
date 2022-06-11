import dataclasses
import typing as tp
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import jax_metrics as jm
import typing_extensions as tpe

import elegy as eg
import elegy.pytree as pytree_m
from elegy import types
from elegy.optimizer import Optimizer

A = tp.TypeVar("A")
M = tp.TypeVar("M", bound="Module")


@tpe.runtime_checkable
class HasOptimizer(tp.Protocol):
    optimizer: tp.Optional[Optimizer]


@tpe.runtime_checkable
class HasLossesAndMetrics(tp.Protocol):
    losses_and_metrics: tp.Optional[jm.LossesAndMetrics]


class ModuleMeta(pytree_m.PytreeObjectMeta):
    def construct(cls, obj: M, *args, **kwargs) -> M:
        obj = super().construct(obj, *args, **kwargs)

        if dataclasses.is_dataclass(obj) and not obj._called_init:
            try:
                super(type(obj), obj).__init__()
            except BaseException as e:
                raise RuntimeError(
                    f"Failed to initialize dataclass Module '{obj.__class__.__name__}'. "
                    f"Dataclass CoreModule's automatically call `super().__init__()` with no arguments but "
                    f"this failed with the following error:"
                ) from e

        if not obj._called_init:
            raise RuntimeError(
                f"CoreModule '{type(obj).__name__}' not properly initialized. "
                "Make sure to call `super().__init__(...)` to propagate initialization."
            )

        return obj


class CoreModule(pytree_m.PytreeObject, metaclass=ModuleMeta):

    initialized: bool = pytree_m.field(pytree_node=False)
    _called_init: bool = pytree_m.field(default=False, pytree_node=False)

    def __init__(
        self,
        *,
        initialized: bool = False,
    ) -> None:

        self.initialized = initialized
        self._called_init = True

    # ---------------------------------------------------------------------------
    # API
    # ---------------------------------------------------------------------------

    def reset_step(
        self: M,
    ) -> M:
        return self

    def init_step(
        self: M,
        key: jnp.ndarray,
        batch: tp.Any,
    ) -> M:
        raise types.MissingMethod()

    def predict_step(
        self: M,
        batch: tp.Any,
        batch_idx: int,
    ) -> tp.Tuple[types.Outputs, M]:
        raise types.MissingMethod()

    def test_step(
        self: M,
        batch: tp.Any,
        batch_idx: int,
    ) -> tp.Tuple[types.Logs, M]:
        raise types.MissingMethod()

    def train_step(
        self: M,
        batch: types.Batch,
        batch_idx: int,
        epoch_idx: int,
    ) -> tp.Tuple[types.Logs, M]:
        raise types.MissingMethod()

    def tabulate(
        self,
        *args,
        summary_depth: int = 2,
        **kwargs,
    ) -> str:
        raise types.MissingMethod()

    # ---------------------------------------------------------------------------
    # Base methods
    # ---------------------------------------------------------------------------

    def set_trainer_params(
        self: M,
        *,
        losses_and_metrics: tp.Optional[jm.LossesAndMetrics] = None,
        optimizer: tp.Optional[Optimizer] = None,
    ) -> M:
        field_updates = {}
        if optimizer is not None:
            if isinstance(self, HasOptimizer):
                if self.optimizer is not None:
                    raise RuntimeError(
                        f"Trying to set optimizer for {type(self)} but it already has one."
                    )
                else:
                    field_updates["optimizer"] = optimizer
            else:
                raise RuntimeError(
                    f"Trying to set optimizer for {type(self)} but has not field 'optimizer'."
                )

        if losses_and_metrics is not None:
            if isinstance(self, HasLossesAndMetrics):
                if self.losses_and_metrics is not None:
                    raise RuntimeError(
                        f"Trying to set losses_and_metrics for {type(self)} but it already has one."
                    )
                else:
                    field_updates["losses_and_metrics"] = losses_and_metrics
            else:
                raise RuntimeError(
                    f"Trying to set losses_and_metrics for {type(self)} but has not field 'losses_and_metrics'."
                )

        self = tp.cast(M, self)

        return self.replace(**field_updates)

    def set_strategy(
        self: M,
        strategy: tp.Union[str, "eg.Strategy"],
    ) -> M:
        raise types.MissingMethod()

    def mark_initialized(
        self: M,
    ) -> M:
        return self.replace(initialized=True)

    # ---------------------------------------------------------------------------
    # Callback Methods
    # ---------------------------------------------------------------------------

    def on_epoch_begin(self: M, epoch: int, logs: tp.Optional[types.Logs] = None) -> M:
        """Called at the end of an epoch.

        Subclasses should override for any actions to run. This function should only
        be called during TRAIN mode.

        Arguments:
            epoch: integer, index of epoch.
            logs: dict, metric results for this training epoch, and for the
                validation epoch if validation is performed. Validation result keys
                are prefixed with `val_`.
        """
        return self

    def on_epoch_end(self: M, epoch: int, logs: tp.Optional[types.Logs] = None) -> M:
        """Called at the end of an epoch.

        Subclasses should override for any actions to run. This function should only
        be called during TRAIN mode.

        Arguments:
            epoch: integer, index of epoch.
            logs: dict, metric results for this training epoch, and for the
                validation epoch if validation is performed. Validation result keys
                are prefixed with `val_`.
        """
        return self

    def on_train_batch_begin(
        self: M, batch: int, logs: tp.Optional[types.Logs] = None
    ) -> M:
        """Called at the beginning of a training batch in `fit` methods.

        Subclasses should override for any actions to run.

        Arguments:
            batch: integer, index of batch within the current epoch.
            logs: dict. Has keys `batch` and `size` representing the current batch
                number and the size of the batch.
        """
        return self

    def on_train_batch_end(
        self: M, batch: int, logs: tp.Optional[types.Logs] = None
    ) -> M:
        """Called at the end of a training batch in `fit` methods.

        Subclasses should override for any actions to run.

        Arguments:
            batch: integer, index of batch within the current epoch.
            logs: dict. Metric results for this batch.
        """
        return self

    def on_test_batch_begin(
        self: M, batch: int, logs: tp.Optional[types.Logs] = None
    ) -> M:
        """Called at the beginning of a batch in `evaluate` methods.

        Also called at the beginning of a validation batch in the `fit`
        methods, if validation data is provided.

        Subclasses should override for any actions to run.

        Arguments:
            batch: integer, index of batch within the current epoch.
            logs: dict. Has keys `batch` and `size` representing the current batch
                number and the size of the batch.
        """
        return self

    def on_test_batch_end(
        self: M, batch: int, logs: tp.Optional[types.Logs] = None
    ) -> M:
        """Called at the end of a batch in `evaluate` methods.

        Also called at the end of a validation batch in the `fit`
        methods, if validation data is provided.

        Subclasses should override for any actions to run.

        Arguments:
            batch: integer, index of batch within the current epoch.
            logs: dict. Metric results for this batch.
        """
        return self

    def on_predict_batch_begin(
        self: M, batch: int, logs: tp.Optional[types.Logs] = None
    ) -> M:
        """Called at the beginning of a batch in `predict` methods.

        Subclasses should override for any actions to run.

        Arguments:
            batch: integer, index of batch within the current epoch.
            logs: dict. Has keys `batch` and `size` representing the current batch
                number and the size of the batch.
        """
        return self

    def on_predict_batch_end(
        self: M, batch: int, logs: tp.Optional[types.Logs] = None
    ) -> M:
        """Called at the end of a batch in `predict` methods.

        Subclasses should override for any actions to run.

        Arguments:
            batch: integer, index of batch within the current epoch.
            logs: dict. Metric results for this batch.
        """
        return self

    def on_train_begin(self: M, logs: tp.Optional[types.Logs] = None) -> M:
        """Called at the beginning of training.

        Subclasses should override for any actions to run.

        Arguments:
            logs: dict. Currently no data is passed to this argument for this method
                but that may change in the future.
        """
        return self

    def on_train_end(self: M, logs: tp.Optional[types.Logs] = None) -> M:
        """Called at the end of training.

        Subclasses should override for any actions to run.

        Arguments:
            logs: dict. Currently no data is passed to this argument for this method
                but that may change in the future.
        """
        return self

    def on_test_begin(self: M, logs: tp.Optional[types.Logs] = None) -> M:
        """Called at the beginning of evaluation or validation.

        Subclasses should override for any actions to run.

        Arguments:
            logs: dict. Currently no data is passed to this argument for this method
                but that may change in the future.
        """
        return self

    def on_test_end(self: M, logs: tp.Optional[types.Logs] = None) -> M:
        """Called at the end of evaluation or validation.

        Subclasses should override for any actions to run.

        Arguments:
            logs: dict. Currently no data is passed to this argument for this method
                but that may change in the future.
        """
        return self

    def on_predict_begin(self: M, logs: tp.Optional[types.Logs] = None) -> M:
        """Called at the beginning of prediction.

        Subclasses should override for any actions to run.

        Arguments:
            logs: dict. Currently no data is passed to this argument for this method
                but that may change in the future.
        """
        return self

    def on_predict_end(self: M, logs: tp.Optional[types.Logs] = None) -> M:
        """Called at the end of prediction.

        Subclasses should override for any actions to run.

        Arguments:
            logs: dict. Currently no data is passed to this argument for this method
                but that may change in the future.
        """
        return self
