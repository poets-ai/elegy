from elegy import utils
import jax
from elegy.losses.loss import Loss, Reduction
import jax.numpy as jnp
import haiku as hk
import typing as tp


class GlobalL1L2Regularization(Loss):
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
        model_fn=model_fn,
        loss=lambda: [elegy.losses.SparseCategoricalCrossentropy()],
        aux_losses=lambda: [elegy.regularizers.GlobalL1L2Regularization(l1=1e-5, l2=1e-4)],
        metrics=lambda: elegy.metrics.SparseCategoricalAccuracy(),
    )
    ```
    
    Attributes:
        l1: L1 regularization factor.
        l2: L2 regularization factor.
    """

    def __init__(
        self,
        l1=0.0,
        l2=0.0,
        reduction: tp.Optional[Reduction] = None,
        name: tp.Optional[str] = None,
        weight: tp.Optional[float] = None,
    ):  # pylint: disable=redefined-outer-name
        super().__init__(reduction=reduction, name=name, weight=weight)

        self.l1 = l1
        self.l2 = l2

    def call(self, params: hk.Params) -> jnp.ndarray:
        """
        Computes the L1 and L2 regularization penalty simultaneously.

        Arguments:
            params: A structure with all the parameters of the model.
        """

        regularization: jnp.ndarray = jnp.array(0.0)

        if not self.l1 and not self.l2:
            return regularization

        if self.l1:
            regularization += self.l1 * sum(
                jnp.sum(jnp.abs(p)) for p in jax.tree_leaves(params)
            )

        if self.l2:
            regularization += self.l2 * sum(
                jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params)
            )

        return regularization

