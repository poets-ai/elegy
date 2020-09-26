from elegy.utils import TrivialPytree
import typing as tp

import jax
import jax.tree_util
import numpy as np
import jax.numpy as jnp


from elegy.types import PRNGKey


@jax.tree_util.register_pytree_node_class
class RNG(TrivialPytree):
    key: jnp.ndarray

    def __init__(self, key: tp.Union[int, jnp.ndarray]):
        self.key = (
            jax.random.PRNGKey(key) if isinstance(key, int) or key.shape == () else key
        )

    def __call__(self) -> np.ndarray:
        self.key = jax.random.split(self.key, 1)[0]
        return self.key

    def __repr__(self) -> str:
        return f"RNG(key={self.key})"
