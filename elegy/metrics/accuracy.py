import typing as tp

import jax.numpy as jnp

from elegy.metrics.mean_metric_wrapper import MeanMetricWrapper


def accuracy(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    # [y_pred, y_true], _ = metrics_utils.ragged_assert_compatible_and_get_flat_values(
    #     [y_pred, y_true]
    # )
    # y_pred.shape.assert_is_compatible_with(y_true.shape)

    if y_true.dtype != y_pred.dtype:
        y_pred = y_pred.astype(y_true.dtype)

    if len(y_pred.shape) > len(y_true.shape):
        y_pred = jnp.argmax(y_pred, axis=-1)

    return (y_true == y_pred).astype(jnp.float32)


class Accuracy(MeanMetricWrapper):
    """
    Calculates how often predictions equals labels. This metric creates two local variables, 
    `total` and `count` that are used to compute the frequency with which `y_pred` matches `y_true`. This frequency is
    ultimately returned as `binary accuracy`: an idempotent operation that simply
    divides `total` by `count`. If `sample_weight` is `None`, weights default to 1. 
    Use `sample_weight` of 0 to mask values.

    ```python
    m = elegy.metrics.Accuracy()

    result = m(y_true=jnp.array([1, 1, 1, 1]), y_pred=jnp.array([0, 1, 1, 1]))
    assert result == 0.75

    result = m(y_true=jnp.array([1, 1, 1, 1]), y_pred=jnp.array([1, 0, 0, 0]))
    assert result == 0.5
    ```

    Usage with elegy API:

    ```python
    model = elegy.Model(
        model_fn,
        loss=lambda: [elegy.losses.SoftmaxCrossentropy()]
        metrics=lambda: [elegy.metrics.Accuracy()]
    )
    ```
    """

    def __init__(self, **kwargs):
        """
        Arguments:
            kwargs: All arguments accepted by `elegy.metrics.Reduce` and `elegy.metrics.Metric`
        """
        super().__init__(accuracy, **kwargs)

