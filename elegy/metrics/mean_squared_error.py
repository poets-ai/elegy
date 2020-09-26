from elegy import types
import typing as tp

import jax.numpy as jnp

from elegy.losses.mean_squared_error import mean_squared_error
from elegy.metrics.mean import Mean


class MeanSquaredError(Mean):
    """
    Computes the cumulative mean squared error between `y_true` and `y_pred`.

    Usage:
    ```python
    mse = elegy.metrics.MeanSquaredError()

    result = mse(y_true=jnp.array([1, 1, 1, 1]), y_pred=jnp.array([0, 1, 1, 1]))
    assert result == 0.25

    result = mse(y_true=jnp.array([1, 1, 1, 1]), y_pred=jnp.array([1, 0, 0, 0]))
    assert result == 0.5
    ```

    Usage with elegy API:

    ```python
    model = elegy.Model(
        module_fn,
        loss=elegy.losses.CategoricalCrossentropy(),
        metrics=elegy.metrics.MeanSquaredError(),
    )
    ```
    """

    def __init__(self, on: tp.Optional[types.IndexLike] = None, **kwargs):
        """
        Creates a `MeanSquaredError` instance.

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
        super().__init__(on=on, **kwargs)

    def call(
        self,
        y_true: jnp.ndarray,
        y_pred: jnp.ndarray,
        sample_weight: tp.Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Accumulates metric statistics. `y_true` and `y_pred` should have the same shape.

        Arguments:
            y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
            y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
            sample_weight: Optional `sample_weight` acts as a
                coefficient for the metric. If a scalar is provided, then the metric is
                simply scaled by the given value. If `sample_weight` is a tensor of size
                `[batch_size]`, then the metric for each sample of the batch is rescaled
                by the corresponding element in the `sample_weight` vector. If the shape
                of `sample_weight` is `[batch_size, d0, .. dN-1]` (or can be broadcasted
                to this shape), then each metric element of `y_pred` is scaled by the
                corresponding value of `sample_weight`. (Note on `dN-1`: all metric
                functions reduce by 1 dimension, usually the last axis (-1)).
        Returns:
            Array with the cumulative accuracy.
        """

        return super().call(values=mean_squared_error(y_true=y_true, y_pred=y_pred))
