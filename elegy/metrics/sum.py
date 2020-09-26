from elegy import types
import typing as tp

import jax.numpy as jnp

from elegy.metrics.reduce import Reduce, Reduction


class Sum(Reduce):
    """
    Computes the (weighted) sum of the given values.

    For example, if values is [1, 3, 5, 7] then the sum is 16.
    If the weights were specified as [1, 1, 0, 0] then the sum would be 4.
    This metric creates one variable, `total`, that is used to compute the sum of
    `values`. This is ultimately returned as `sum`.
    If `sample_weight` is `None`, weights default to 1.  Use `sample_weight` of 0
    to mask values.

    Usage:
    ```python
    m = elegy.metrics.Sum()
    assert 16.0 == m([1, 3, 5, 7])
    ```
    Usage with Elegy API:
    ```python
    model = elegy.Model(
        module_fn,
        loss=elegy.losses.CategoricalCrossentropy(),
        metrics=elegy.metrics.Sum(name='sum_1'),
    )
    model = elegy.Model(inputs, outputs)
    ```
    """

    def __init__(self, on: tp.Optional[types.IndexLike] = None, **kwargs):
        """Creates a `Sum` instance.

        Arguments:
            on: A string or integer, or iterable of string or integers, that
                indicate how to index/filter the `y_true` and `y_pred`
                arguments before passing them to `call`. For example if `on = "a"` then
                `y_true = y_true["a"]`. If `on` is an iterable
                the structures will be indexed iteratively, for example if `on = ["a", 0, "b"]`
                then `y_true = y_true["a"][0]["b"]`, same for `y_pred`. For more information
                check out [Keras-like behavior](https://poets-ai.github.io/elegy/guides/modules-losses-metrics/#keras-like-behavior).
            kwargs: Additional keyword arguments passed to Module.
        """
        super().__init__(reduction=Reduction.SUM, on=on, **kwargs)
