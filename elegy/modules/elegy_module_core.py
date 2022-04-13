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


class CoreModule(to.Tree, to.Immutable, to.Map, to.Copy):

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

    # ---------------------------------------------------------------------------
    # API
    # ---------------------------------------------------------------------------

    def reset_metrics(
        self: M,
    ) -> M:
        raise types.MissingMethod()

    def init_on_batch(
        self: M,
        key: jnp.ndarray,
        inputs: tp.Any,
    ) -> M:
        raise types.MissingMethod()

    def predict_on_batch(
        self: M,
        inputs: types.Inputs,
    ) -> tp.Tuple[types.Outputs, M]:
        raise types.MissingMethod()

    def test_on_batch(
        self: M,
        inputs: types.Inputs,
        labels: types.Labels,
    ) -> tp.Tuple[types.Logs, M]:
        raise types.MissingMethod()

    def train_on_batch(
        self: M,
        inputs: types.Inputs,
        labels: types.Labels,
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
