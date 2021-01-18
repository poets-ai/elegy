import sys
import typing as tp
from copy import copy
from enum import Enum
from functools import total_ordering

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np

from elegy.frozen_dict import FrozenDict


if sys.version_info >= (3, 8):
    from typing import Protocol, runtime_checkable
else:
    from typing_extensions import Protocol, runtime_checkable

EPSILON = 1e-7


class Mode(str, Enum):
    pred = "pred"
    test = "test"
    train = "train"


class TrivialPytree:
    def tree_flatten(self):
        return tuple(vars(self).values()), None

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return cls(*children)


class Empty:
    pass


class ModuleOrderError(Exception):
    pass


EMPTY = Empty()


@jax.tree_util.register_pytree_node_class
class Uninitialized:
    def tree_flatten(self):
        return ((), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls()


UNINITIALIZED = Uninitialized()


@jax.tree_util.register_pytree_node_class
class RNGSeq(TrivialPytree):
    key: jnp.ndarray

    def __init__(self, key: tp.Union[int, jnp.ndarray]):
        self.key = (
            jax.random.PRNGKey(key) if isinstance(key, int) or key.shape == () else key
        )

    def next(self) -> jnp.ndarray:
        self.key = jax.random.split(self.key, 1)[0]
        return self.key

    def __repr__(self) -> str:
        return f"RNGSeq(key={self.key})"


T = tp.TypeVar("T")
Container = tp.Union[
    T,
    tp.Tuple["Container", ...],
    tp.Dict[str, "Container"],
]
ArrayHolder = tp.Union[Container[np.ndarray], np.ndarray]

IndexLike = tp.Union[str, int, tp.Iterable[tp.Union[str, int]]]

Shape = tp.Sequence[int]
ShapeLike = tp.Union[int, Shape]
FloatLike = tp.Union[float, np.ndarray]
DType = tp.Any
ParamName = str

Params = tp.Mapping[str, tp.Mapping[ParamName, np.ndarray]]
State = tp.Mapping[str, tp.Mapping[str, np.ndarray]]
PadFn = tp.Callable[[int], tp.Tuple[int, int]]
PadFnOrFns = tp.Union[PadFn, tp.Sequence[PadFn]]
PRNGKey = np.ndarray
Parameters = tp.Any
ParameterCollection = tp.Dict[str, Parameters]
Logs = tp.Dict[str, tp.Union[np.ndarray, float]]
Index = tp.Union[int, str]
Path = tp.Tuple[Index, ...]
Grads = tp.Any
RNG = tp.Union[RNGSeq, np.ndarray]
Scalar = tp.Union[np.ndarray, float]
SummaryModule = tp.Any
SummaryValue = tp.Any
Summaries = tp.List[
    tp.Tuple[Path, tp.Optional[SummaryModule], SummaryValue],
]


class OutputStates(tp.NamedTuple):
    preds: tp.Any
    params: tp.Any
    states: tp.Any


class States(tp.NamedTuple):
    net_params: tp.Any = UNINITIALIZED
    net_states: tp.Any = UNINITIALIZED
    metrics_states: tp.Any = UNINITIALIZED
    optimizer_states: tp.Any = UNINITIALIZED
    rng: tp.Union[RNG, tp.Any] = UNINITIALIZED

    def update(
        self,
        net_params: tp.Any = UNINITIALIZED,
        net_states: tp.Any = UNINITIALIZED,
        metrics_states: tp.Any = UNINITIALIZED,
        optimizer_states: tp.Any = UNINITIALIZED,
        rng: tp.Union[RNG, Uninitialized] = UNINITIALIZED,
    ) -> "States":

        updates = {}

        if not isinstance(net_params, Uninitialized):
            updates["net_params"] = net_params
        if not isinstance(net_states, Uninitialized):
            updates["net_states"] = net_states
        if not isinstance(metrics_states, Uninitialized):
            updates["metrics_states"] = metrics_states
        if not isinstance(optimizer_states, Uninitialized):
            updates["optimizer_states"] = optimizer_states
        if not isinstance(rng, Uninitialized):
            updates["rng"] = rng

        kwargs = {field: getattr(self, field) for field in self._fields}
        kwargs.update(**updates)

        return States(**kwargs)

    def merge_new(self, other: "States") -> "States":

        updates = {}

        if isinstance(self.net_params, Uninitialized) and not isinstance(
            other.net_params, Uninitialized
        ):
            updates["net_params"] = other.net_params

        if isinstance(self.net_states, Uninitialized) and not isinstance(
            other.net_states, Uninitialized
        ):
            updates["net_states"] = other.net_states

        if isinstance(self.metrics_states, Uninitialized) and not isinstance(
            other.metrics_states, Uninitialized
        ):
            updates["metrics_states"] = other.metrics_states

        if isinstance(self.optimizer_states, Uninitialized) and not isinstance(
            other.optimizer_states, Uninitialized
        ):
            updates["optimizer_states"] = other.optimizer_states

        if isinstance(self.rng, Uninitialized) and not isinstance(
            other.rng, Uninitialized
        ):
            updates["rng"] = other.rng

        kwargs = {field: getattr(self, field) for field in self._fields}
        kwargs.update(**updates)

        return States(**kwargs)

    def merge(self, other: "States") -> "States":
        return other.update(*other)

    def copy(self) -> "States":
        return States(
            net_params=self.net_params,
            net_states=self.net_states,
            metrics_states=self.metrics_states,
            optimizer_states=self.optimizer_states,
            rng=copy(self.rng),
        )


class Prediction(tp.NamedTuple):
    pred: tp.Any
    states: States


class Evaluation(tp.NamedTuple):
    loss: Scalar
    logs: Logs
    states: States


class Backprop(tp.NamedTuple):
    loss: Scalar
    logs: Logs
    states: States
    grads: Grads


class Training(tp.NamedTuple):
    logs: Logs
    states: States


class Initializer(Protocol):
    def __call__(self, shape: Shape, dtype: DType) -> np.ndarray:
        ...


class MissingModule(Exception):
    pass


class MissingOptimizer(Exception):
    pass


class MissingMethod(Exception):
    pass
