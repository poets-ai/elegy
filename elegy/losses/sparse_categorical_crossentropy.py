from elegy import types
import typing as tp

import jax
import jax.numpy as jnp

from elegy import utils
from elegy.losses.loss import Loss, Reduction
from elegy.losses.categorical_crossentropy import categorical_crossentropy


def sparse_categorical_crossentropy(
    y_true: jnp.ndarray, y_pred: jnp.ndarray, from_logits: bool = False
) -> jnp.ndarray:

    if from_logits:
        y_pred = jax.nn.log_softmax(y_pred)
        return -jnp.take_along_axis(y_pred, y_true[..., None], axis=-1)[..., 0]

    else:
        # select output value
        y_pred = jnp.take_along_axis(y_pred, y_true[..., None], axis=-1)[..., 0]

        # calculate log
        y_pred = jnp.maximum(y_pred, utils.EPSILON)
        y_pred = jnp.log(y_pred)
        return -y_pred


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

    Usage with the `Elegy` API:

    ```python
    model = elegy.Model(
        module_fn,
        loss=lelegy.losses.SparseCategoricalCrossentropy(),
        metrics=lelegy.metrics.Accuracy(),
        optimizer=optax.adam(1e-3),
    )

    ```
    """

    def __init__(
        self,
        from_logits: bool = False,
        reduction: tp.Optional[Reduction] = None,
        weight: tp.Optional[float] = None,
        on: tp.Optional[types.IndexLike] = None,
        **kwargs
    ):
        """
        Initializes `SparseCategoricalCrossentropy` instance.

        Arguments:
            from_logits: Whether `y_pred` is expected to be a logits tensor. By
                default, we assume that `y_pred` encodes a probability distribution.
                **Note - Using from_logits=True is more numerically stable.**
            reduction: (Optional) Type of `elegy.losses.Reduction` to apply to
                loss. Default value is `SUM_OVER_BATCH_SIZE`. For almost all cases
                this defaults to `SUM_OVER_BATCH_SIZE`.
            weight: Optional weight contribution for the total loss. Defaults to `1`.
            on: A string or integer, or iterable of string or integers, that
                indicate how to index/filter the `y_true` and `y_pred`
                arguments before passing them to `call`. For example if `on = "a"` then
                `y_true = y_true["a"]`. If `on` is an iterable
                the structures will be indexed iteratively, for example if `on = ["a", 0, "b"]`
                then `y_true = y_true["a"][0]["b"]`, same for `y_pred`. For more information
                check out [Keras-like behavior](https://poets-ai.github.io/elegy/guides/modules-losses-metrics/#keras-like-behavior).
        """
        super().__init__(reduction=reduction, weight=weight, on=on, **kwargs)

        self._from_logits = from_logits

    def call(
        self, y_true, y_pred, sample_weight: tp.Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Invokes the `SparseCategoricalCrossentropy` instance.

        Arguments:
            y_true: Ground truth values.
            y_pred: The predicted values.
            sample_weight: Acts as a
                coefficient for the loss. If a scalar is provided, then the loss is
                simply scaled by the given value. If `sample_weight` is a tensor of size
                `[batch_size]`, then the total loss for each sample of the batch is
                rescaled by the corresponding element in the `sample_weight` vector. If
                the shape of `sample_weight` is `[batch_size, d0, .. dN-1]` (or can be
                broadcasted to this shape), then each loss element of `y_pred` is scaled
                by the corresponding value of `sample_weight`. (Note on`dN-1`: all loss
                functions reduce by 1 dimension, usually axis=-1.)

        Returns:
            Loss values per sample.
        """

        return sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=self._from_logits
        )
