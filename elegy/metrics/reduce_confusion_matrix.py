import typing as tp
from enum import Enum

import jax.numpy as jnp
import numpy as np

from elegy import initializers, module, types, utils, module
from elegy.metrics.metric import Metric


class Reduction(Enum):
    TRUE_POSITIVES = "true_positives"
    FALSE_POSITIVES = "false_positives"
    FALSE_NEGATIVES = "false_negatives"
    TRUE_NEGATIVES = "true_negatives"


def reduce(
    cm_metric: jnp.ndarray,
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
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
            sample_weight = jnp.broadcast_to(sample_weight, y_true.shape)
        except ValueError as e:
            raise e

    if reduction == Reduction.TRUE_POSITIVES:
        mask = y_pred == 1
        hits = y_true == y_pred
        if sample_weight is not None:
            hits = hits * sample_weight
        value = jnp.sum(hits * mask)

    if reduction == Reduction.FALSE_POSITIVES:
        mask = y_pred == 1
        misses = y_true != y_pred
        if sample_weight is not None:
            misses = misses * sample_weight
        value = jnp.sum(misses * mask)

    if reduction == Reduction.FALSE_NEGATIVES:
        mask = y_true == 1
        misses = y_true != y_pred
        if sample_weight is not None:
            misses = misses * sample_weight
        value = jnp.sum(misses * mask)

    if reduction == Reduction.TRUE_NEGATIVES:
        mask = y_true == 0
        hits = y_true == y_pred
        if sample_weight is not None:
            hits = hits * sample_weight
        value = jnp.sum(hits * mask)

    cm_metric += value
    return cm_metric


class ReduceConfusionMatrix(Metric):
    """Encapsulates confusion matrix metrics that perform a reduce operation on the values."""

    def __init__(
        self, reduction: Reduction, on: tp.Optional[types.IndexLike] = None, **kwargs
    ):
        super().__init__(on=on, **kwargs)

        self._reduction = reduction

        if self._reduction not in (
            Reduction.TRUE_POSITIVES,
            Reduction.FALSE_POSITIVES,
            Reduction.FALSE_NEGATIVES,
            Reduction.TRUE_NEGATIVES,
        ):
            raise NotImplementedError(
                "reduction {reduction} not implemented".format(
                    reduction=self._reduction
                )
            )

    def call(
        self,
        y_true: jnp.ndarray,
        y_pred: jnp.ndarray,
        sample_weight: tp.Optional[jnp.ndarray] = None,
    ) -> tp.Any:
        """
        Accumulates confusion matrix metrics for computing the reduction metric.

        Arguments:
            y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.

            y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

            sample_weight: Optional weighting of each example. Defaults to 1.

        Returns:
            Array with the cumulative reduce metric.
        """

        cm_metric = self.add_parameter(
            "cm_metric",
            lambda: jnp.array(0, dtype=jnp.int32),
            trainable=False,
        )

        cm_metric = reduce(
            cm_metric=cm_metric,
            y_true=y_true,
            y_pred=y_pred,
            reduction=self._reduction,
            sample_weight=sample_weight,
            dtype=self.dtype,
        )

        self.update_parameter("cm_metric", cm_metric)

        return cm_metric
