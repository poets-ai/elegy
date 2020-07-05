import typing as tp

import jax.numpy as jnp

from elegy.metrics.mean import Mean
from elegy import utils


class MetricFn(utils.Protocol):
    def __call__(
        self, y_true: jnp.ndarray, y_pred: jnp.ndarray, **kwargs
    ) -> jnp.ndarray:
        ...


class MeanMetricWrapper(Mean):
    """Wraps a stateless metric function with the Mean metric."""

    def __init__(
        self,
        fn: MetricFn,
        name: tp.Optional[str] = None,
        dtype: tp.Optional[jnp.dtype] = None,
        **kwargs
    ):
        """Creates a `MeanMetricWrapper` instance.
    Args:
      fn: The metric function to wrap, with signature
        `fn(y_true, y_pred, **kwargs)`.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
      **kwargs: The keyword arguments that are passed on to `fn`.
    """
        super().__init__(name=name, dtype=dtype)
        self._fn = fn
        self._fn_kwargs = kwargs

    @utils.inject_dependencies
    def call(
        self,
        y_true: jnp.ndarray,
        y_pred: jnp.ndarray,
        sample_weight: tp.Optional[jnp.ndarray] = None,
    ):
        """Accumulates metric statistics.
    `y_true` and `y_pred` should have the same shape.
    Args:
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
      Update op.
    """
        y_true = y_true.astype(self._dtype)
        y_pred = y_pred.astype(self._dtype)

        # (
        #     [y_true, y_pred],
        #     sample_weight,
        # ) = metrics_utils.ragged_assert_compatible_and_get_flat_values(
        #     [y_true, y_pred], sample_weight
        # )

        # y_pred, y_true = tf_losses_utils.squeeze_or_expand_dimensions(y_pred, y_true)

        matches = self._fn(y_true, y_pred, **self._fn_kwargs)

        return super().call(matches, sample_weight=sample_weight)

