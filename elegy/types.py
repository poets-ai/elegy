import typing as tp
import numpy as np
import jax.numpy as jnp

ArrayLike = tp.Union[jnp.ndarray, np.ndarray]
T = tp.TypeVar("T")
Container = tp.Union[
    T, tp.Tuple["Container", ...], tp.Dict[str, "Container"],
]
ArrayHolder = Container[ArrayLike]
