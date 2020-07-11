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
        model_fn,
        loss=lambda: [elegy.losses.CategoricalCrossentropy()],
        metrics=lambda: [elegy.metrics.MeanSquaredError()],
    )
    ```
    """

    def __init__(
        self, name: tp.Optional[str] = None, dtype: tp.Optional[jnp.dtype] = None,
    ):
        """
        Creates a `MeanSquaredError` instance.

        Arguments:
            name: string name of the metric instance.
            dtype: data type of the metric result.
        """
        super().__init__(name=name, dtype=dtype)

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
