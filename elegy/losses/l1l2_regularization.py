from elegy import utils
import jax
from elegy.losses.loss import Loss, Reduction
import jax.numpy as jnp
import haiku as hk
import typing as tp


class GlobalL1L2Regularization(Loss):
    r"""A regularizer that applies both L1 and L2 regularization penalties.
  The L1 regularization penalty is computed as:
  $$\ell_1\,\,penalty =\ell_1\sum_{i=0}^n|x_i|$$
  The L2 regularization penalty is computed as
  $$\ell_2\,\,penalty =\ell_2\sum_{i=0}^nx_i^2$$
  Attributes:
      l1: Float; L1 regularization factor.
      l2: Float; L2 regularization factor.
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

    def call(self, params: hk.Params):

        if not self.l1 and not self.l2:
            return jnp.array(0.0)

        regularization = 0.0

        if self.l1:
            regularization += self.l1 * sum(
                jnp.sum(jnp.abs(p)) for p in jax.tree_leaves(params)
            )

        if self.l2:
            regularization += self.l2 * sum(
                jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params)
            )

        return regularization


def GlobalL1Regularization(
    l=0.01, reduction: tp.Optional[Reduction] = None, name: str = "l1_regularization",
) -> GlobalL1L2Regularization:
    r"""
    Create a regularizer that applies an L1 regularization penalty.
  
    The L1 regularization penalty is computed as:
    
    $$\ell_1\,\,penalty =\ell_1\sum_{i=0}^n|x_i|$$

    Arguments:
        l: Float; L1 regularization factor.
    
    Returns:
        An L1 Regularizer with the given regularization factor.
  """
    return GlobalL1L2Regularization(l1=l, reduction=reduction, name=name)


def GlobalL2Regularization(
    l=0.01, reduction: tp.Optional[Reduction] = None, name="l2_regularization",
) -> GlobalL1L2Regularization:
    r"""
    Create a regularizer that applies an L2 regularization penalty.
  
    The L2 regularization penalty is computed as:
    
    $$\ell_2\,\,penalty =\ell_2\sum_{i=0}^nx_i^2$$
    
    Arguments:
        l: Float; L2 regularization factor.
  
    Returns:
        An L2 Regularizer with the given regularization factor.
  """
    return GlobalL1L2Regularization(l2=l, reduction=reduction, name=name)

