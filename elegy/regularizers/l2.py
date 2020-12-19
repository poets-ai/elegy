import typing as tp

from elegy.regularizers.l1l2 import L1L2


class L2(L1L2):
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
            elegy.losses.GlobaL2Regularization(l=1e-4),
        ],
        metrics=lambda: elegy.metrics.SparseCategoricalAccuracy(),
    )
    ```
    """

    def __init__(self, l: float = 0.01, **kwargs):
        r"""
        Creates an L2 instance.

        Arguments:
            l: L2 regularization factor.
        """
        super().__init__(l2=l, **kwargs)
