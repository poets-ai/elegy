import typing as tp
from enum import Enum

import jax.numpy as jnp
import numpy as np

from elegy import initializers, module, types, utils
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
    sample_weight: tp.Optional[np.ndarray],
    dtype: jnp.dtype,
) -> tp.Tuple[jnp.ndarray, jnp.ndarray, tp.Optional[jnp.ndarray]]:

    if sample_weight is not None:
        sample_weight = sample_weight.astype(dtype)

        # Update dimensions of weights to match with values if possible.
        # values, _, sample_weight = tf_losses_utils.squeeze_or_expand_dimensions(
        #     values, sample_weight=sample_weight
        # )

        try:
            # Broadcast weights if possible.
            sample_weight = jnp.broadcast_to(sample_weight, values.shape)
        except ValueError:
            # Reduce values to same ndim as weight array
            ndim = values.ndim
            weight_ndim = sample_weight.ndim
            if reduction == Reduction.SUM:
                values = jnp.sum(values, axis=list(range(weight_ndim, ndim)))
            else:
                values = jnp.mean(values, axis=list(range(weight_ndim, ndim)))

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
            num_values = jnp.prod(jnp.array(values.shape)).astype(dtype)
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

    def __init__(self, reduction: Reduction, **kwargs):
        super().__init__(**kwargs)

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

    def call(
        self, values: jnp.ndarray, sample_weight: tp.Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Accumulates statistics for computing the reduction metric. For example, if `values` is [1, 3, 5, 7]
        and reduction=SUM_OVER_BATCH_SIZE, then the value of `result()` is 4. If the `sample_weight`
        is specified as [1, 1, 0, 0] then value of `result()` would be 2.

        Arguments:
            values: Per-example value.
            sample_weight: Optional weighting of each example. Defaults to 1.

        Returns:
            Array with the cumulative reduce.
        """
        total = self.add_parameter(
            "total",
            lambda: jnp.array(0, dtype=jnp.int32),
            trainable=False,
        )

        if self._reduction in (
            Reduction.SUM_OVER_BATCH_SIZE,
            Reduction.WEIGHTED_MEAN,
        ):
            count = self.add_parameter(
                "count",
                lambda: jnp.array(0, dtype=jnp.int32),
                trainable=False,
            )
        else:
            count = None

        value, total, count = reduce(
            total=total,
            count=count,
            values=values,
            reduction=self._reduction,
            sample_weight=sample_weight,
            dtype=self.dtype,
        )

        self.update_parameter("total", total)

        if count is not None:
            self.update_parameter("count", count)

        return value
