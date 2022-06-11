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

EPSILON = 1e-7
F = tp.TypeVar("F", bound=tp.Callable)

A = tp.TypeVar("A")
B = tp.TypeVar("B")
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
Outputs = tp.Any
Loss = jnp.ndarray
Batch = tp.Any
Index = tp.Union[int, str]
Path = tp.Tuple[Index, ...]
Grads = tp.Any
RNG = tp.Union["KeySeq", np.ndarray]
Scalar = tp.Union[np.ndarray, float, int]
SummaryModule = tp.Any
SummaryValue = tp.Any
KeyLike = tp.Union[int, jnp.ndarray]

NetParams = tp.Any
NetStates = tp.Any
ModuleParams = tp.Any
ModuleStates = tp.Any
MetricsStates = tp.Any
OptimizerStates = tp.Any
OptimizerStates = tp.Any
Grads = tp.Any
Pytree = tp.Any


def Key(seed: tp.Union[int, jnp.ndarray]) -> jnp.ndarray:
    return jax.random.PRNGKey(seed) if isinstance(seed, int) else seed


class KeySeq:
    """KeySeq is simple module that can produce a sequence of PRNGKeys.

    Example:
    ```python
    class Dropout(Module):
        rng: KeySeq()

        def __init__(self, rate: float):
            self.next_key = KeySeq()
            ...

        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            key = self.next_key()
            mask = jax.random.bernoulli(key, 1.0 - self.rate)
            ...
    ```
    """

    key: jnp.ndarray

    def __init__(
        self,
        key: tp.Union[jnp.ndarray, int],
    ):
        """
        Arguments:
            key: An optional PRNGKey to initialize the KeySeq with.
        """

        self.key = Key(key)

    def next(self) -> jnp.ndarray:
        """
        Return a new PRNGKey and updates the internal rng state.

        Returns:
            A PRNGKey.
        """

        key, self.key = jax.random.split(self.key)

        return key

    __next__ = next


class Hashable(tp.Generic[A]):
    """A hashable immutable wrapper around non-hashable values"""

    value: A

    def __init__(self, value: A):
        self.__dict__["value"] = value

    def __setattr__(self, name: str, value: tp.Any) -> None:
        raise AttributeError(f"Hashable is immutable")


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
