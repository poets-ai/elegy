import typing as tp
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import treeo as to
import treex as tx

from elegy import types

A = tp.TypeVar("A")
M = tp.TypeVar("M", bound="CoreModule")
Me = tp.TypeVar("Me", bound="tx.Metric")


InitStep = tp.Callable[[M, jnp.ndarray, types.Inputs], M]
PredStep = tp.Callable[[M, types.Inputs], tp.Tuple[types.Outputs, M]]
TestStep = tp.Callable[[M, types.Inputs, types.Labels], tp.Tuple[types.Logs, M]]
TrainStep = tp.Callable[[M, types.Inputs, types.Labels], tp.Tuple[types.Logs, M]]


class CoreModuleMeta(to.TreeMeta):
    def construct(cls, obj: M, *args, **kwargs) -> M:
        obj = super().construct(obj, *args, **kwargs)

        if not hasattr(obj, "_called_init"):
            CoreModule.__init__(obj)

        return obj


class CoreModule(to.Tree, to.Immutable, to.Map, to.Copy, metaclass=CoreModuleMeta):

    initialized: bool
    distributed_strategy: "DistributedStrategy"

    def __init__(
        self,
        *,
        initialized: bool = False,
        distributed_strategy: tp.Optional["DistributedStrategy"] = None,
    ) -> None:
        from elegy.distributed_strategies import Eager

        self.initialized = initialized
        self.distributed_strategy = (
            distributed_strategy if distributed_strategy is not None else Eager()
        )
        self._called_init = None

    # ---------------------------------------------------------------------------
    # API
    # ---------------------------------------------------------------------------

    def reset_step(
        self: M,
    ) -> M:
        raise types.MissingMethod()

    def init_step(
        self: M,
        key: jnp.ndarray,
        inputs: tp.Any,
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
        batch: types.Inputs,
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

    def mark_initialized(
        self: M,
    ) -> M:
        return self.replace(initialized=True)

    def set_distributed_strategy(
        self: M,
        distributed_strategy: "DistributedStrategy",
    ) -> M:
        return self.replace(distributed_strategy=distributed_strategy)

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


class DistributedStrategy(ABC):
    def from_local(self, module: M) -> M:
        return module

    def to_local(self, module: M) -> M:
        return module

    def lift_data(self, data: A) -> A:
        return data

    def lift_key(self, key: jnp.ndarray) -> jnp.ndarray:
        return key

    def lift_batch_size(self, batch_size: int) -> int:
        return batch_size

    def handle_post_init(self, module: M) -> M:
        return module

    def handle_metrics(
        self,
        metrics: Me,
    ) -> Me:
        return metrics

    def handle_mudule_and_grads(
        self,
        module: M,
        grads: tp.Any,
    ) -> tp.Tuple[M, tp.Any]:
        return module, grads

    @abstractmethod
    def init_step_fn(self, module: M) -> InitStep[M]:
        ...

    @abstractmethod
    def pred_step_fn(self, module: M) -> PredStep[M]:
        ...

    @abstractmethod
    def test_step_fn(self, module: M) -> TestStep[M]:
        ...

    @abstractmethod
    def train_step_fn(self, module: M) -> TrainStep[M]:
        ...

    # implement order methods, required so that DistributedStrategy can be
    # used as a key in a dict
    def __lt__(self, other):
        return self.__class__.__name__ < other.__class__.__name__

    def __le__(self, other):
        return self.__class__.__name__ <= other.__class__.__name__

    def __gt__(self, other):
        return self.__class__.__name__ > other.__class__.__name__

    def __ge__(self, other):
        return self.__class__.__name__ >= other.__class__.__name__
