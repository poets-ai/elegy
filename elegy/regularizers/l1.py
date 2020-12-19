import typing as tp

from elegy.regularizers.l1l2 import L1L2


class L1(L1L2):
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
            elegy.regularizers.L1(l=1e-5)
        ],
        metrics=lambda: elegy.metrics.SparseCategoricalAccuracy(),
    )
    ```
    """

    def __init__(self, l: float = 0.01, **kwargs):
        r"""
        Create a L1 instance.

        Arguments:
            l: L1 regularization factor.
            kwargs: Additional keyword arguments passed to Module.
        """
        return super().__init__(l1=l, **kwargs)
