from elegy import utils
import typing as tp

import jax.numpy as jnp

from elegy.metrics.mean import Mean
from elegy.metrics.accuracy import accuracy


def categorical_accuracy(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:

    y_true = jnp.argmax(y_true, axis=-1)
    y_pred = jnp.argmax(y_pred, axis=-1)

    return accuracy(y_true, y_pred)


class CategoricalAccuracy(Mean):
    """
    Calculates how often predictions matches one-hot labels.
    You can provide logits of classes as `y_pred`, since argmax of
    logits and probabilities are same.
    
    This metric creates two local variables, `total` and `count` that are used to
    compute the frequency with which `y_pred` matches `y_true`. This frequency is
    ultimately returned as `categorical accuracy`: an idempotent operation that
    simply divides `total` by `count`.
    
    `y_pred` and `y_true` should be passed in as vectors of probabilities, rather
    than as labels. If necessary, use `tf.one_hot` to expand `y_true` as a vector.
    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.
    
    Usage:

    ```python
    m = elegy.metrics.CategoricalAccuracy()

    result = m(
        y_true=jnp.array([[0, 0, 1], [0, 1, 0]]), 
        y_pred=jnp.array([[0.1, 0.9, 0.8], [0.05, 0.95, 0]]),
    )
    assert result == 0.5  # 1/2

    result = m(
        y_true=jnp.array([[0, 1, 0], [0, 1, 0]]), 
        y_pred=jnp.array([[0.1, 0.9, 0.8], [0.05, 0.95, 0]]),
    )
    assert result ==  0.75  # 3/4
    ```

    Usage with elegy API:
    ```python
    model = elegy.Model(
        model_fn,
        loss=lambda: [elegy.losses.CategoricalCrossentropy()],
        metrics=lambda: [elegy.metrics.CategoricalAccuracy()])
        optimizer=optix.adam(1e-3),
    )
    ```
    """

    def __init__(
        self, name: tp.Optional[str] = None, dtype: tp.Optional[jnp.dtype] = None,
    ):
        """
        Creates a `CategoricalAccuracy` instance.
        
        Arguments:
            name: string name of the metric instance.
            dtype: data type of the metric result.
        """
        super().__init__(name=name, dtype=dtype)

    @utils.inject_dependencies
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

        return super().call(
            values=categorical_accuracy(y_true=y_true, y_pred=y_pred),
            sample_weight=sample_weight,
        )

