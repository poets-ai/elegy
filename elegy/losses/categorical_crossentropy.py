import typing as tp

import jax
import jax.numpy as jnp

from elegy import utils
from elegy.losses.loss import Loss, Reduction


def categorical_crossentropy(
    y_true: jnp.ndarray, y_pred: jnp.ndarray, from_logits: bool = False
) -> jnp.ndarray:

    # if self._sparse_labels:
    #     y_true = jax.nn.one_hot(y_true, y_pred.shape[-1])

    if from_logits:
        y_pred = jax.nn.log_softmax(y_pred)

    else:
        y_pred = jnp.maximum(y_pred, utils.EPSILON)
        y_pred = jnp.log(y_pred)

    return -jnp.sum(y_true * y_pred, axis=-1)


class CategoricalCrossentropy(Loss):
    """
    Computes the crossentropy loss between the labels and predictions.
    Use this crossentropy loss function when there are two or more label classes.
    We expect labels to be provided in a `one_hot` representation. If you want to
    provide labels as integers, please use `SparseCategoricalCrossentropy` loss.
    There should be `# classes` floating point values per feature.
    In the snippet below, there is `# classes` floating pointing values per
    example. The shape of both `y_pred` and `y_true` are
    `[batch_size, num_classes]`.
    
    Usage:
    ```python
    y_true = jnp.array([[0, 1, 0], [0, 0, 1]])
    y_pred = jnp.array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])

    # Using 'auto'/'sum_over_batch_size' reduction type.
    cce = elegy.losses.CategoricalCrossentropy()

    assert cce(y_true, y_pred) == 1.177
    # Calling with 'sample_weight'.
    assert cce(y_true, y_pred, sample_weight=tf.constant([0.3, 0.7])) == 0.814
    # Using 'sum' reduction type.
    cce = elegy.losses.CategoricalCrossentropy(
        reduction=elegy.losses.Reduction.SUM
    )
    assert cce(y_true, y_pred) == 2.354
    # Using 'none' reduction type.
    cce = elegy.losses.CategoricalCrossentropy(
        reduction=elegy.losses.Reduction.NONE
    )

    assert list(cce(y_true, y_pred)) == [0.0513, 2.303]
    ```

    Usage with the `compile` API:
    ```python
    model = elegy.Model(
        model_fn,
        loss=lambda: [elegy.losses.CategoricalCrossentropy()]
        metrics=lambda: [elegy.metrics.Accuracy()]
        optimizer=optix.adam(1e-3),
    )
    ```
    """

    def __init__(
        self,
        from_logits: bool = False,
        label_smoothing: float = 0,
        reduction: tp.Optional[Reduction] = None,
        name: tp.Optional[str] = None,
        weight: tp.Optional[float] = None,
    ):
        """Initializes `CategoricalCrossentropy` instance.
        Arguments:
        from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, we assume that `y_pred` encodes a probability distribution.
            **Note - Using from_logits=True is more numerically stable.**
        label_smoothing: Float in [0, 1]. When > 0, label values are smoothed,
            meaning the confidence on label values are relaxed. e.g.
            `label_smoothing=0.2` means that we will use a value of `0.1` for label
            `0` and `0.9` for label `1`"
        reduction: (Optional) Type of `elegy.losses.Reduction` to apply to
            loss. Default value is `AUTO`. `AUTO` indicates that the reduction
            option will be determined by the usage context. For almost all cases
            this defaults to `SUM_OVER_BATCH_SIZE`. When used with
            `tf.distribute.Strategy`, outside of built-in training loops such as
            `elegy` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
            will raise an error. Please see this custom training [tutorial]
            (https://www.tensorflow.org/tutorials/distribute/custom_training)
            for more details.
        name: Optional name for the op. Defaults to 'categorical_crossentropy'.
        """
        super().__init__(reduction=reduction, name=name, weight=weight)

        self._from_logits = from_logits
        self._label_smoothing = label_smoothing

    def call(
        self, y_true, y_pred, sample_weight: tp.Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:

        return categorical_crossentropy(y_true, y_pred, from_logits=self._from_logits)
