from elegy import types
import typing as tp

import jax.numpy as jnp

from elegy.losses.binary_crossentropy import binary_crossentropy
from elegy.metrics.mean import Mean


class BinaryCrossentropy(Mean):
    """
    Computes the crossentropy metric between the labels and predictions.
    This is the crossentropy metric class to be used when there are only two
    label classes (0 and 1).

    Usage:
    ```python
    y_true=jnp.array([[0., 1.], [0., 0.]]),
    y_pred=jnp.array([[0.6, 0.4], [0.4, 0.6]])
    
    bce = elegy.metrics.BinaryCrossentropy()
    result = bce(
        y_true=y_true,
        y_pred=y_pred,
    )
    assert jnp.isclose(result, 0.815, rtol=0.01)

    # BCE using sample_weight
    bce = elegy.metrics.BinaryCrossentropy()
    result = bce(y_true, y_pred, sample_weight=jnp.array([1., 0.]))
    assert jnp.isclose(result, 0.916, rtol=0.01)
    ```

    Usage with elegy API:

    ```python
    model = elegy.Model(
        module_fn,
        loss=elegy.losses.CategoricalCrossentropy(),
        metrics=elegy.metrics.BinaryCrossentropy.defer(),
    )
    ```
    """

    def __init__(
        self,
        name: tp.Optional[str] = None,
        from_logits: bool = False,
        dtype: tp.Optional[jnp.dtype] = None,
        on: tp.Optional[types.IndexLike] = None,
    ):
        """Creates a `BinaryCrossentropy` instance.
        Args:
        name: string name of the metric instance.
        dtype: data type of the metric result.
        """

        super().__init__(name=name, dtype=dtype, on=on)
        self._from_logits = from_logits

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
                from_logits: True if the predicted data are logits instead of probabilities
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
            values=binary_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=self._from_logits), 
            sample_weight=sample_weight
        )