import typing as tp

import jax.numpy as jnp
import numpy as np

from elegy import types, utils
from elegy.metrics.accuracy import accuracy
from elegy.metrics.mean import Mean


def binary_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """
    Calculates how often predictions matches binary labels.

    Standalone usage:

    ```python
    y_true = np.array([[1], [1], [0], [0]])
    y_pred = np.array([[1], [1], [0], [0]])
    m = elegy.metrics.binary_accuracy(y_true, y_pred)
    assert m.shape == (4,)
    ```
    Arguments:
        y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
        threshold: Float representing the threshold for deciding whether
            prediction values are 1 or 0.

    Returns:
        Binary accuracy values. shape = `[batch_size, d0, .. dN-1]`
    """
    assert abs(y_pred.ndim - y_true.ndim) <= 1

    y_true, y_pred = utils.maybe_expand_dims(y_true, y_pred)

    y_pred = y_pred > threshold
    return jnp.mean(y_true == y_pred, axis=-1)


class BinaryAccuracy(Mean):
    """
    Calculates how often predictions matches binary labels.

    Standalone usage:
    ```python
    m = elegy.metrics.BinaryAccuracy()
    result = m(
        y_true=np.array([[1], [1], [0], [0]]),
        y_pred=np.array([[0.98], [1], [0], [0.6]]),
    )
    assert result == 0.75

    m = elegy.metrics.BinaryAccuracy()
    result = m(
        y_true=np.array([[1], [1], [0], [0]]),
        y_pred=np.array([[0.98], [1], [0], [0.6]]),
        sample_weight=np.array([1, 0, 0, 1]),
    )
    assert result == 0.5
    ```
    Usage with `Model` API:
    ```python
    model = elegy.Model(
        ...
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )
    ```
    """

    def __init__(
        self, threshold: float = 0.5, on: tp.Optional[types.IndexLike] = None, **kwargs
    ):
        """
        Creates a `BinaryAccuracy` instance.

        Arguments:
            threshold: Float representing the threshold for deciding whether
                prediction values are 1 or 0.
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
        self.threshold = threshold

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
            values=binary_accuracy(
                y_true=y_true, y_pred=y_pred, threshold=self.threshold
            ),
            sample_weight=sample_weight,
        )
