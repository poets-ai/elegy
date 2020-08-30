import typing as tp

import numpy as np

from elegy import utils

ArrayLike = tp.Union[np.ndarray, np.ndarray]
T = tp.TypeVar("T")
Container = tp.Union[
    T,
    tp.Tuple["Container", ...],
    tp.Dict[str, "Container"],
]
ArrayHolder = Container[ArrayLike]

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
States = tp.Dict[str, tp.Any]


class Initializer(utils.Protocol):
    def __call__(self, shape: Shape, dtype: DType) -> np.ndarray:
        ...
