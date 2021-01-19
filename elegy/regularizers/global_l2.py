import typing as tp

from elegy.losses.loss import Reduction
from elegy.regularizers.global_l1l2 import GlobalL1L2


def GlobalL2(
    l: float = 0.01,
    reduction: tp.Optional[Reduction] = None,
    name: str = "l2_regularization",
) -> GlobalL1L2:
    r"""
    Create a regularizer that applies an L2 regularization penalty.

    The L2 regularization penalty is computed as:

    $$\ell_2\,\,penalty =\ell_2\sum_{i=0}^nx_i^2$$

    Usage:

    ```python
    model = elegy.Model(
        module_fn,
        loss=[
            elegy.losses.SparseCategoricalCrossentropy(),
            elegy.losses.GlobalL2Regularization(l=1e-4),
        ],
        metrics=lambda: elegy.metrics.SparseCategoricalAccuracy(),
    )
    ```

    Arguments:
        l: L2 regularization factor.

    Returns:
        An L2 Regularizer with the given regularization factor.
    """
    return GlobalL1L2(l2=l, reduction=reduction, name=name)
