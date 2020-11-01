from elegy import types
from elegy import utils
import typing as tp

import jax.numpy as jnp

from elegy.metrics.mean import Mean
from elegy.metrics.reduce_confusion_matrix import Reduction, ReduceConfusionMatrix
from elegy.metrics.metric import Metric


def precision(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    threshold: jnp.ndarray,
    class_id: jnp.ndarray,
    sample_weight: jnp.ndarray,
    true_positives: ReduceConfusionMatrix,
    false_positives: ReduceConfusionMatrix,
) -> jnp.ndarray:

    # TODO: class_id behavior
    y_pred = (y_pred > threshold).astype(jnp.float32)

    if y_true.dtype != y_pred.dtype:
        y_pred = y_pred.astype(y_true.dtype)

    true_positives = true_positives(
        y_true=y_true, y_pred=y_pred, sample_weight=sample_weight
    )
    false_positives = false_positives(
        y_true=y_true, y_pred=y_pred, sample_weight=sample_weight
    )

    return jnp.nan_to_num(jnp.divide(true_positives, true_positives + false_positives))


class Precision(Metric):
    """
    The metric creates two local variables, `true_positives` and `false_positives`
    that are used to compute the precision. This value is ultimately returned as
    `precision`, an idempotent operation that simply divides `true_positives`
    by the sum of `true_positives` and `false_positives`.

    If `sample_weight` is `None`, weights default to 1. Use `sample_weight` of 0 to mask values.

    If sample_weight is None, weights default to 1. Use sample_weight of 0 to mask values.

    If class_id is specified, we calculate precision by considering only the entries in the batch
    for which class_id is above the threshold and computing the fraction of them for which class_id
    is indeed a correct label.

    ```python
        precision = elegy.metrics.Precision()

        result = precision(
            y_true=jnp.array([0, 1, 1, 1]), y_pred=jnp.array([1, 0, 1, 1])
        )
        assert result == 0.6666667 # 2 / 3

        result = precision(
            y_true=jnp.array([1, 1, 1, 1]), y_pred=jnp.array([1, 1, 0, 0])
        )
        assert result == 0.8 # 4 / 5
    ```

    Usage with elegy API:

    ```python
        model = elegy.Model(
        module_fn,
        loss=elegy.losses.CategoricalCrossentropy(),
        metrics=elegy.metrics.Precision(),
        optimizer=optix.adam(1e-3),
    )
    ```
    """

    def __init__(
        self,
        on: tp.Optional[types.IndexLike] = None,
        threshold=None,
        class_id=None,
        **kwargs
    ):
        """
        Creates a `Precision` instance.

        Arguments:
            on: A string or integer, or iterable of string or integers, that
                indicate how to index/filter the `y_true` and `y_pred`
                arguments before passing them to `call`. For example if `on = "a"` then
                `y_true = y_true["a"]`. If `on` is an iterable
                the structures will be indexed iteratively, for example if `on = ["a", 0, "b"]`
                then `y_true = y_true["a"][0]["b"]`, same for `y_pred`. For more information
                check out [Keras-like behavior](https://poets-ai.github.io/elegy/guides/modules-losses-metrics/#keras-like-behavior).

            threshold: (Optional) A float value or a python list/tuple of float threshold
                values in [0, 1]. A threshold is compared with prediction values to determine
                the truth value of predictions (i.e., above the threshold is true, below is false).
                One metric value is generated for each threshold value. If neither threshold is set
                the default is to calculate precision with threshold=0.5.

            class_id: (Optional) Integer class ID for which we want binary metrics.
                This must be in the half-open interval `[0, num_classes)`, where
                `num_classes` is the last dimension of predictions.

            kwargs: Additional keyword arguments passed to Module.
        """
        super().__init__(on=on, **kwargs)
        self.threshold = 0.5 if threshold is None else threshold
        self.class_id = 1 if class_id is None else class_id
        self.true_positives = ReduceConfusionMatrix(reduction=Reduction.TRUE_POSITIVES)
        self.false_positives = ReduceConfusionMatrix(
            reduction=Reduction.FALSE_POSITIVES
        )

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

            sample_weight: Optional weighting of each example. Defaults to 1. Can be a
                `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
                be broadcastable to `y_true`.
        Returns:
            Array with the cumulative precision.
        """

        return precision(
            y_true=y_true,
            y_pred=y_pred,
            sample_weight=sample_weight,
            threshold=self.threshold,
            class_id=self.class_id,
            true_positives=self.true_positives,
            false_positives=self.false_positives,
        )
