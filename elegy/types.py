import typing as tp
import numpy as np
import jax.numpy as jnp

ArrayLike = tp.Union[jnp.ndarray, np.ndarray]
T = tp.TypeVar("T")
Container = tp.Union[
    T, tp.Tuple["Container", ...], tp.Dict[str, "Container"],
]
ArrayHolder = Container[ArrayLike]

IndexLike = tp.Union[str, int, tp.Iterable[tp.Union[str, int]]]

Shape = tp.Sequence[int]
ShapeLike = tp.Union[int, Shape]
FloatLike = tp.Union[float, jnp.ndarray]
DType = tp.Any
ParamName = str
Initializer = tp.Callable[[Shape, DType], jnp.ndarray]
Params = tp.Mapping[str, tp.Mapping[ParamName, jnp.ndarray]]
State = tp.Mapping[str, tp.Mapping[str, jnp.ndarray]]
PadFn = tp.Callable[[int], tp.Tuple[int, int]]
PadFnOrFns = tp.Union[PadFn, tp.Sequence[PadFn]]
