import typing as tp

from elegy.losses.loss import Reduction
from elegy.regularizers.global_l1l2_regularization import GlobalL1L2Regularization


def GlobalL2Regularization(
    l: float = 0.01,
    reduction: tp.Optional[Reduction] = None,
    name: str = "l2_regularization",
) -> GlobalL1L2Regularization:
    r"""
    Create a regularizer that applies an L2 regularization penalty.
  
    The L2 regularization penalty is computed as:
    
    $$\ell_2\,\,penalty =\ell_2\sum_{i=0}^nx_i^2$$

    Usage:

    ```python
    model = elegy.Model(
        model_fn,
        loss=lambda: [elegy.losses.SparseCategoricalCrossentropy()],
        aux_losses=lambda: [elegy.losses.GlobaL2Regularization(l=1e-4)],
        metrics=lambda: elegy.metrics.SparseCategoricalAccuracy(),
    )
    ```
    
    Arguments:
        l: L2 regularization factor.
  
    Returns:
        An L2 Regularizer with the given regularization factor.
    """
    return GlobalL1L2Regularization(l2=l, reduction=reduction, name=name)
