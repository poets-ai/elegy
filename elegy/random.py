import typing as tp

import jax
import numpy as np

from elegy.types import PRNGKey


class RNG:
    key: np.ndarray

    def __init__(self, key: tp.Union[int, np.ndarray]):
        self.key = (
            jax.random.PRNGKey(key) if isinstance(key, int) or key.shape == () else key
        )

    def __call__(self) -> np.ndarray:
        self.key = jax.random.split(self.key, 1)
        return self.key
