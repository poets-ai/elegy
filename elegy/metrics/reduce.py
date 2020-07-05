from elegy import utils
from enum import Enum
import typing as tp

import haiku as hk
import jax.numpy as jnp
from elegy.metrics.metric import Metric


class Reduction(Enum):
    SUM = "sum"
    SUM_OVER_BATCH_SIZE = "sum_over_batch_size"
    WEIGHTED_MEAN = "weighted_mean"


def reduce(
    total: jnp.ndarray,
    count: tp.Optional[jnp.ndarray],
    values: jnp.ndarray,
    reduction: Reduction,
    sample_weight: tp.Optional[jnp.ndarray],
    dtype: jnp.dtype,
):

    if sample_weight is not None:
        sample_weight = sample_weight.astype(dtype)

        # Update dimensions of weights to match with values if possible.
        # values, _, sample_weight = tf_losses_utils.squeeze_or_expand_dimensions(
        #     values, sample_weight=sample_weight
        # )

        # try:
        #     # Broadcast weights if possible.
        #     sample_weight = weights_broadcast_ops.broadcast_weights(
        #         sample_weight, values
        #     )
        # except ValueError:
        #     # Reduce values to same ndim as weight array
        #     ndim = K.ndim(values)
        #     weight_ndim = K.ndim(sample_weight)
        #     if reduction == metrics_utils.Reduction.SUM:
        #         values = math_ops.reduce_sum(
        #             values, axis=list(range(weight_ndim, ndim))
        #         )
        #     else:
        #         values = math_ops.reduce_mean(
        #             values, axis=list(range(weight_ndim, ndim))
        #         )
        values = values * sample_weight

    value_sum = jnp.sum(values)

    total += value_sum

    # Exit early if the reduction doesn't have a denominator.
    if reduction == Reduction.SUM:
        num_values = None

    # Update `count` for reductions that require a denominator.
    elif reduction == Reduction.SUM_OVER_BATCH_SIZE:
        num_values = jnp.prod(values.shape).astype(dtype)

    else:
        if sample_weight is None:
            num_values = jnp.prod(values.shape).astype(dtype)
        else:
            num_values = jnp.sum(sample_weight)

    if count is not None and num_values is not None:
        count += num_values

    if reduction == Reduction.SUM:
        value = total
    else:
        value = total / count

    return value, total, count


class Reduce(Metric):
    """Encapsulates metrics that perform a reduce operation on the values."""

    def __init__(
        self,
        reduction: Reduction,
        name: tp.Optional[str] = None,
        dtype: tp.Optional[jnp.dtype] = None,
    ):
        super().__init__(name=name, dtype=dtype)

        self._reduction = reduction

        if self._reduction not in (
            Reduction.SUM,
            Reduction.SUM_OVER_BATCH_SIZE,
            Reduction.WEIGHTED_MEAN,
        ):
            raise NotImplementedError(
                "reduction {reduction} not implemented".format(
                    reduction=self._reduction
                )
            )

    @utils.inject_dependencies
    def call(self, values, sample_weight: tp.Optional[jnp.ndarray] = None):
        """Accumulates statistics for computing the reduction metric.
    For example, if `values` is [1, 3, 5, 7] and reduction=SUM_OVER_BATCH_SIZE,
    then the value of `result()` is 4. If the `sample_weight` is specified as
    [1, 1, 0, 0] then value of `result()` would be 2.
    Args:
      values: Per-example value.
      sample_weight: Optional weighting of each example. Defaults to 1.
    Returns:
      Update op.
    """
        total = hk.get_state(
            "total", shape=[], dtype=self._dtype, init=hk.initializers.Constant(0)
        )

        if self._reduction in (Reduction.SUM_OVER_BATCH_SIZE, Reduction.WEIGHTED_MEAN,):
            count = hk.get_state(
                "count", shape=[], dtype=jnp.int64, init=hk.initializers.Constant(0)
            )
        else:
            count = None

        value, total, count = reduce(
            total=total,
            count=count,
            values=values,
            reduction=self._reduction,
            sample_weight=sample_weight,
            dtype=self._dtype,
        )

        hk.set_state("total", total)

        if count is not None:
            hk.set_state("count", count)

        return value

