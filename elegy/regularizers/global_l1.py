import typing as tp

from elegy.losses.loss import Reduction
from elegy.regularizers.global_l1l2 import GlobalL1L2


def GlobalL1(
    l: float = 0.01,
    reduction: tp.Optional[Reduction] = None,
    name: str = "l1_regularization",
    **kwargs
) -> GlobalL1L2:
    r"""
    Create a regularizer that applies an L1 regularization penalty.

    The L1 regularization penalty is computed as:

    $$\ell_1\,\,penalty =\ell_1\sum_{i=0}^n|x_i|$$

    Usage:

    ```python
    model = elegy.Model(
        module_fn,
        loss=[
            elegy.losses.SparseCategoricalCrossentropy(),
            elegy.regularizers.GlobalL1(l=1e-5)
        ],
        metrics=lambda: elegy.metrics.SparseCategoricalAccuracy(),
    )
    ```

    Arguments:
        l: L1 regularization factor.
        kwargs: Additional keyword arguments passed to Module.

    Returns:
        An L1 Regularizer with the given regularization factor.
    """
    return GlobalL1L2(l1=l, reduction=reduction, name=name, **kwargs)
