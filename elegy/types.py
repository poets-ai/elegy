from copy import copy
import typing as tp

import numpy as np

from elegy import utils

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
Parameters = tp.Dict[str, tp.Any]
Logs = tp.Dict[str, tp.Union[np.ndarray, float]]
RNG = tp.Union[utils.RNGSeq, np.ndarray]
Scalar = tp.Union[np.ndarray, float]


class OutputStates(tp.NamedTuple):
    preds: tp.Any
    params: tp.Any
    states: tp.Any


class States(tp.NamedTuple):
    net_params: tp.Any = utils.UNINITIALIZED
    net_states: tp.Any = utils.UNINITIALIZED
    metrics_states: tp.Any = utils.UNINITIALIZED
    optimizer_states: tp.Any = utils.UNINITIALIZED
    rng: tp.Union[RNG, tp.Any] = utils.UNINITIALIZED

    def update(
        self,
        net_params: tp.Any = utils.UNINITIALIZED,
        net_states: tp.Any = utils.UNINITIALIZED,
        metrics_states: tp.Any = utils.UNINITIALIZED,
        optimizer_states: tp.Any = utils.UNINITIALIZED,
        rng: tp.Union[RNG, utils.Uninitialized] = utils.UNINITIALIZED,
    ) -> "States":

        updates = {}

        if not isinstance(net_params, utils.Uninitialized):
            updates["net_params"] = net_params
        if not isinstance(net_states, utils.Uninitialized):
            updates["net_states"] = net_states
        if not isinstance(metrics_states, utils.Uninitialized):
            updates["metrics_states"] = metrics_states
        if not isinstance(optimizer_states, utils.Uninitialized):
            updates["optimizer_states"] = optimizer_states
        if not isinstance(rng, utils.Uninitialized):
            updates["rng"] = rng

        kwargs = {field: getattr(self, field) for field in self._fields}
        kwargs.update(**updates)

        return States(**kwargs)

    def merge_new(self, other: "States") -> "States":

        updates = {}

        if isinstance(self.net_params, utils.Uninitialized) and not isinstance(
            other.net_params, utils.Uninitialized
        ):
            updates["net_params"] = other.net_params

        if isinstance(self.net_states, utils.Uninitialized) and not isinstance(
            other.net_states, utils.Uninitialized
        ):
            updates["net_states"] = other.net_states

        if isinstance(self.metrics_states, utils.Uninitialized) and not isinstance(
            other.metrics_states, utils.Uninitialized
        ):
            updates["metrics_states"] = other.metrics_states

        if isinstance(self.optimizer_states, utils.Uninitialized) and not isinstance(
            other.optimizer_states, utils.Uninitialized
        ):
            updates["optimizer_states"] = other.optimizer_states

        if isinstance(self.rng, utils.Uninitialized) and not isinstance(
            other.rng, utils.Uninitialized
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
    logs: Logs
    states: States


class Initializer(utils.Protocol):
    def __call__(self, shape: Shape, dtype: DType) -> np.ndarray:
        ...
