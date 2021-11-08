import sys
import typing as tp
from copy import copy
from dataclasses import dataclass, field
from enum import Enum
from functools import total_ordering

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np
import treex as tx

EPSILON = 1e-7
F = tp.TypeVar("F", bound=tp.Callable)


KeySeq = tx.KeySeq

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
Labels = tp.Mapping[str, tp.Any]
ParameterCollection = tp.Dict[str, Parameters]
Logs = tp.Dict[str, jnp.ndarray]
Index = tp.Union[int, str]
Path = tp.Tuple[Index, ...]
Grads = tp.Any
RNG = tp.Union[KeySeq, np.ndarray]
Scalar = tp.Union[np.ndarray, float, int]
SummaryModule = tp.Any
SummaryValue = tp.Any

NetParams = tp.Any
NetStates = tp.Any
ModuleParams = tp.Any
ModuleStates = tp.Any
MetricsStates = tp.Any
OptimizerStates = tp.Any
OptimizerStates = tp.Any
Grads = tp.Any
Pytree = tp.Any


class MissingModule(Exception):
    pass


class MissingOptimizer(Exception):
    pass


class MissingMethod(Exception):
    pass


class DependencyUnavailable(Exception):
    pass


class ShapeMismatch(Exception):
    pass


class MissingParameter(Exception):
    pass


class NoContext(Exception):
    pass


class ModuleOrderError(Exception):
    pass


class SubmoduleNotRegistered(Exception):
    pass


class ModelNotInitialized(Exception):
    pass
