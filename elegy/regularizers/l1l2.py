import typing as tp

import haiku as hk
import jax
import jax.numpy as jnp

from elegy import utils
from elegy.regularizers.regularizer import Regularizer


class L1L2(Regularizer):
    r"""
    A regularizer that applies both L1 and L2 regularization penalties.

    The L1 regularization penalty is computed as:

    $$
    \ell_1\,\,penalty =\ell_1\sum_{i=0}^n|x_i|
    $$

    The L2 regularization penalty is computed as

    $$\ell_2\,\,penalty =\ell_2\sum_{i=0}^nx_i^2$$


    Usage:

    ```python
    model = elegy.Model(
        module_fn,
        loss=[
            elegy.losses.SparseCategoricalCrossentropy(),
            elegy.regularizers.L1L2(l1=1e-5, l2=1e-4),
        ],
        metrics=lambda: elegy.metrics.SparseCategoricalAccuracy(),
    )
    ```

    Attributes:
        l1: L1 regularization factor.
        l2: L2 regularization factor.
    """

    def __init__(
        self, l1=0.0, l2=0.0, **kwargs
    ):  # pylint: disable=redefined-outer-name
        super().__init__(**kwargs)
        self.l1 = l1
        self.l2 = l2

    def call(self, parameters: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the L1 and L2 regularization penalty simultaneously.

        Arguments:
            parameters: A structure with all the parameters of the model.
        """

        regularization: jnp.ndarray = jnp.array(0.0)

        if not self.l1 and not self.l2:
            return regularization

        if self.l1:
            regularization += self.l1 * jnp.sum(jnp.abs(parameters))

        if self.l2:
            regularization += self.l2 * jnp.sum(jnp.square(parameters))

        return regularization
