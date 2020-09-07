import typing as tp
from enum import Enum

import jax.numpy as jnp
import numpy as np

from elegy import initializers, module, types, utils, hooks
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
        if sample_weight is not None:
            y_true = y_true * sample_weight
            y_pred = y_pred * sample_weight
        mask = y_pred == 1
        value = jnp.sum(y_true[mask] == 1)

    if reduction == Reduction.FALSE_POSITIVES:
        if sample_weight is not None:
            y_true = y_true * sample_weight
            y_pred = y_pred * sample_weight
        mask = y_pred == 1
        value = jnp.sum(y_true[mask] == 0)

    if reduction == Reduction.FALSE_NEGATIVES:
        if sample_weight is not None:
            y_true = y_true * sample_weight
            y_pred = y_pred * sample_weight
        mask = y_true == 1
        value = jnp.sum(y_pred[mask] == 0)

    if reduction == Reduction.TRUE_NEGATIVES:
        if sample_weight is not None:
            y_true = y_true * sample_weight
            y_pred = y_pred * sample_weight
        mask = y_true == 0
        value = jnp.sum(y_pred[mask] == 0)

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
            Reduction.FALSE_NEGATIVES,
            Reduction.FALSE_POSITIVES,
            Reduction.TRUE_POSITIVES,
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

        if self._reduction in (
            Reduction.FALSE_NEGATIVES,
            Reduction.FALSE_POSITIVES,
            Reduction.TRUE_POSITIVES,
            Reduction.TRUE_NEGATIVES,
        ):
            cm_metric = hooks.get_state(
                self._reduction.value,
                shape=[],
                dtype=jnp.int32,
                initializer=initializers.Constant(0),
            )
        else:
            count = None

        cm_metric = reduce(
            cm_metric=cm_metric,
            y_true=y_true,
            y_pred=y_pred,
            reduction=self._reduction,
            sample_weight=sample_weight,
            dtype=self.dtype,
        )

        if cm_metric is not None:
            hooks.set_state(self._reduction.value, cm_metric)

        return cm_metric
