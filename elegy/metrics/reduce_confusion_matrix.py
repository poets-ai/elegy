import typing as tp
from enum import Enum

import jax
import jax.numpy as jnp
import numpy as np

from elegy import initializers, module, types, utils, module
from elegy.metrics.metric import Metric


class Reduction(Enum):
    TRUE_POSITIVES = "true_positives"
    FALSE_POSITIVES = "false_positives"
    FALSE_NEGATIVES = "false_negatives"
    TRUE_NEGATIVES = "true_negatives"

    MULTICLASS_TRUE_POSITIVES = "multiclass_true_positives"
    MULTICLASS_FALSE_POSITIVES = "multiclass_false_positives"
    MULTICLASS_FALSE_NEGATIVES = "multiclass_false_negatives"


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

    if reduction == Reduction.MULTICLASS_TRUE_POSITIVES:
        hits = (y_true == y_pred).astype(jnp.float32)
        if sample_weight is not None:
            hits = hits * sample_weight
        scan_op = lambda carry, x: (None, (hits*(y_true==x)).sum())
        _, value = jax.lax.scan(scan_op, None, jnp.arange(cm_metric.shape[0]))

    if reduction == Reduction.MULTICLASS_FALSE_POSITIVES:
        misses = (y_true != y_pred).astype(jnp.float32)
        if sample_weight is not None:
            misses = misses * sample_weight
        scan_op = lambda carry, x: (None, (misses*(y_pred==x)).sum())
        _, value = jax.lax.scan(scan_op, None, jnp.arange(cm_metric.shape[0]))

    if reduction == Reduction.MULTICLASS_FALSE_NEGATIVES:
        misses = (y_true != y_pred).astype(jnp.float32)
        if sample_weight is not None:
            misses = misses * sample_weight
        scan_op = lambda carry, x: (None, (misses*(y_true==x)).sum())
        _, value = jax.lax.scan(scan_op, None, jnp.arange(cm_metric.shape[0]))

    cm_metric += value
    return cm_metric


class ReduceConfusionMatrix(Metric):
    """Encapsulates confusion matrix metrics that perform a reduce operation on the values."""

    def __init__(
        self,
        reduction: Reduction,
        on: tp.Optional[types.IndexLike] = None,
        n_classes: tp.Optional[int] = None,
        **kwargs,
    ):
        super().__init__(on=on, **kwargs)

        self._reduction = reduction
        self._n_classes = n_classes

        if self._reduction not in (
            Reduction.TRUE_POSITIVES,
            Reduction.FALSE_POSITIVES,
            Reduction.FALSE_NEGATIVES,
            Reduction.TRUE_NEGATIVES,
            Reduction.MULTICLASS_TRUE_POSITIVES,
            Reduction.MULTICLASS_FALSE_POSITIVES,
            Reduction.MULTICLASS_FALSE_NEGATIVES,
        ):
            raise NotImplementedError(
                "reduction {reduction} not implemented".format(
                    reduction=self._reduction
                )
            )

        if "multiclass" in self._reduction.value and self._n_classes is None:
            raise ValueError(
                f"Argument `n_classes` required for reduction {self._reduction}"
            )

    def call(
        self,
        y_true: jnp.ndarray,
        y_pred: jnp.ndarray,
        sample_weight: tp.Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Accumulates confusion matrix metrics for computing the reduction metric.

        Arguments:
            y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.

            y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

            sample_weight: Optional weighting of each example. Defaults to 1.

        Returns:
            Array with the cummulative reduce metric.
        """

        cm_metric = self.add_parameter(
            "cm_metric",
            shape=[] if self._n_classes is None else [self._n_classes],
            dtype=jnp.int32,
            initializer=initializers.Constant(0),
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
