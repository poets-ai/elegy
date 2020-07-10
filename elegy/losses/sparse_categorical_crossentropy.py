import typing as tp

import jax
import jax.numpy as jnp

from elegy import utils
from elegy.losses.loss import Loss, Reduction
from elegy.losses.categorical_crossentropy import categorical_crossentropy


def sparse_categorical_crossentropy(
    y_true: jnp.ndarray, y_pred: jnp.ndarray, from_logits: bool = False
) -> jnp.ndarray:

    y_true = jax.nn.one_hot(y_true, y_pred.shape[-1])

    return categorical_crossentropy(y_true, y_pred, from_logits=from_logits)


class SparseCategoricalCrossentropy(Loss):
    """
    Computes the crossentropy loss between the labels and predictions.

    Use this crossentropy loss function when there are two or more label classes.
    We expect labels to be provided as integers. If you want to provide labels
    using `one-hot` representation, please use `CategoricalCrossentropy` loss.
    There should be `# classes` floating point values per feature for `y_pred`
    and a single floating point value per feature for `y_true`.
    In the snippet below, there is a single floating point value per example for
    `y_true` and `# classes` floating pointing values per example for `y_pred`.
    The shape of `y_true` is `[batch_size]` and the shape of `y_pred` is
    `[batch_size, num_classes]`.

    Usage:
    ```python
    y_true = jnp.array([1, 2])
    y_pred = jnp.array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])

    # Using 'auto'/'sum_over_batch_size' reduction type.
    scce = elegy.losses.SparseCategoricalCrossentropy()
    result = scce(y_true, y_pred)  # 1.177
    assert jnp.isclose(result, 1.177, rtol=0.01)

    # Calling with 'sample_weight'.
    result = scce(y_true, y_pred, sample_weight=jnp.array([0.3, 0.7]))  # 0.814
    assert jnp.isclose(result, 0.814, rtol=0.01)

    # Using 'sum' reduction type.
    scce = elegy.losses.SparseCategoricalCrossentropy(
        reduction=elegy.losses.Reduction.SUM
    )
    result = scce(y_true, y_pred)  # 2.354
    assert jnp.isclose(result, 2.354, rtol=0.01)

    # Using 'none' reduction type.
    scce = elegy.losses.SparseCategoricalCrossentropy(
        reduction=elegy.losses.Reduction.NONE
    )
    result = scce(y_true, y_pred)  # [0.0513, 2.303]
    assert jnp.all(jnp.isclose(result, [0.0513, 2.303], rtol=0.01))
    ```

    Usage with the `compile` API:
    
    ```python
    model = elegy.Model(
        model_fn,
        loss=lambda: [elegy.losses.SparseCategoricalCrossentropy()]
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
        """Initializes `SparseCategoricalCrossentropy` instance.
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
            name: Optional name for the op. Defaults to 'sparse_categorical_crossentropy'.
            weight: Optional weight contribution for the total loss. Defaults to `1`.
        """
        super().__init__(reduction=reduction, name=name, weight=weight)

        self._from_logits = from_logits
        self._label_smoothing = label_smoothing

    def call(
        self, y_true, y_pred, sample_weight: tp.Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:

        return sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=self._from_logits
        )
