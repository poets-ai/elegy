from elegy import types
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
    accuracy = elegy.metrics.CategoricalAccuracy()

    result = accuracy(
        y_true=jnp.array([[0, 0, 1], [0, 1, 0]]),
        y_pred=jnp.array([[0.1, 0.9, 0.8], [0.05, 0.95, 0]]),
    )
    assert result == 0.5  # 1/2

    result = accuracy(
        y_true=jnp.array([[0, 1, 0], [0, 1, 0]]),
        y_pred=jnp.array([[0.1, 0.9, 0.8], [0.05, 0.95, 0]]),
    )
    assert result ==  0.75  # 3/4
    ```

    Usage with elegy API:
    ```python
    model = elegy.Model(
        module_fn,
        loss=elegy.losses.CategoricalCrossentropy(),
        metrics=elegy.metrics.CategoricalAccuracy(),
        optimizer=optax.adam(1e-3),
    )
    ```
    """

    def __init__(self, on: tp.Optional[types.IndexLike] = None, **kwargs):
        """
        Creates a `CategoricalAccuracy` instance.

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

        return super().call(
            values=categorical_accuracy(y_true=y_true, y_pred=y_pred),
            sample_weight=sample_weight,
        )
