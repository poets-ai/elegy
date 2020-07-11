import typing as tp

from elegy.losses.loss import Reduction
from elegy.regularizers.global_l1l2_regularization import GlobalL1L2Regularization


def GlobalL1Regularization(
    l: float = 0.01,
    reduction: tp.Optional[Reduction] = None,
    name: str = "l1_regularization",
) -> GlobalL1L2Regularization:
    r"""
    Create a regularizer that applies an L1 regularization penalty.
  
    The L1 regularization penalty is computed as:
    
    $$\ell_1\,\,penalty =\ell_1\sum_{i=0}^n|x_i|$$

    Usage:

    ```python
    model = elegy.Model(
        model_fn=model_fn,
        loss=lambda: [elegy.losses.SparseCategoricalCrossentropy()],
        aux_losses=lambda: [elegy.regularizers.GlobalL1Regularization(l=1e-5)],
        metrics=lambda: elegy.metrics.SparseCategoricalAccuracy(),
    )
    ```

    Arguments:
        l: L1 regularization factor.
    
    Returns:
        An L1 Regularizer with the given regularization factor.
  """
    return GlobalL1L2Regularization(l1=l, reduction=reduction, name=name)
