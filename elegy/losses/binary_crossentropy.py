from elegy import types
import typing as tp
import jax
import jax.numpy as jnp
from elegy import utils, types
from elegy.losses.loss import Loss, Reduction


def binary_crossentropy(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    from_logits: bool = False,
    label_smoothing: float = 0,
) -> jnp.ndarray:
    assert abs(y_pred.ndim - y_true.ndim) <= 1

    y_true, y_pred = utils.maybe_expand_dims(y_true, y_pred)

    if label_smoothing:
        y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

    if from_logits:
        return -jnp.mean(y_true * y_pred - jnp.logaddexp(0.0, y_pred), axis=-1)

    y_pred = jnp.clip(y_pred, types.EPSILON, 1.0 - types.EPSILON)
    return -jnp.mean(
        y_true * jnp.log(y_pred) + (1 - y_true) * jnp.log(1 - y_pred), axis=-1
    )


class BinaryCrossentropy(Loss):
    """
    Computes the cross-entropy loss between true labels and predicted labels.
    Use this cross-entropy loss when there are only two label classes (assumed to
    be 0 and 1). For each example, there should be a single floating-point value
    per prediction.
    In the snippet below, each of the four examples has only a single
    floating-pointing value, and both `y_pred` and `y_true` have the shape
    `[batch_size]`.

    Usage:
    ```python
    y_true = jnp.array([[0., 1.], [0., 0.]])
    y_pred = jnp.array[[0.6, 0.4], [0.4, 0.6]])

    # Using 'auto'/'sum_over_batch_size' reduction type.
    bce = elegy.losses.BinaryCrossentropy()
    result = bce(y_true, y_pred)
    assert jnp.isclose(result, 0.815, rtol=0.01)

    # Calling with 'sample_weight'.
    bce = elegy.losses.BinaryCrossentropy()
    result = bce(y_true, y_pred, sample_weight=jnp.array([1, 0]))
    assert jnp.isclose(result, 0.458, rtol=0.01)

    # Using 'sum' reduction type.
    bce = elegy.losses.BinaryCrossentropy(reduction=elegy.losses.Reduction.SUM)
    result = bce(y_true, y_pred)
    assert jnp.isclose(result, 1.630, rtol=0.01)

    # Using 'none' reduction type.
    bce = elegy.losses.BinaryCrossentropy(reduction=elegy.losses.Reduction.NONE)
    result = bce(y_true, y_pred)
    assert jnp.all(jnp.isclose(result, [0.916, 0.713], rtol=0.01))
    ```


    Usage with the `Elegy` API:
    ```python
    model = elegy.Model(
        module_fn,
        loss=elegy.losses.BinaryCrossentropy(),
        metrics=elegy.metrics.Accuracy(),
        optimizer=optax.adam(1e-3),
    )
    ```
    """

    def __init__(
        self,
        from_logits: bool = False,
        label_smoothing: float = 0,
        reduction: tp.Optional[Reduction] = None,
        weight: tp.Optional[float] = None,
        on: tp.Optional[types.IndexLike] = None,
        **kwargs
    ):
        """
        Initializes `CategoricalCrossentropy` instance.

        Arguments:
            from_logits: Whether `y_pred` is expected to be a logits tensor. By
                default, we assume that `y_pred` encodes a probability distribution.
                **Note - Using from_logits=True is more numerically stable.**
            label_smoothing: Float in [0, 1]. When > 0, label values are smoothed,
                meaning the confidence on label values are relaxed. e.g.
                `label_smoothing=0.2` means that we will use a value of `0.1` for label
                `0` and `0.9` for label `1`"
            reduction: (Optional) Type of `elegy.losses.Reduction` to apply to
                loss. Default value is `SUM_OVER_BATCH_SIZE`. Indicates that the reduction
                option will be determined by the usage context. For almost all cases
                this defaults to `SUM_OVER_BATCH_SIZE`. When used with
                `tf.distribute.Strategy`, outside of built-in training loops such as
                `elegy` `compile` and `fit`, ` or `SUM_OVER_BATCH_SIZE`
                will raise an error.
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
        self._label_smoothing = label_smoothing

    def call(
        self,
        y_true: jnp.ndarray,
        y_pred: jnp.ndarray,
        sample_weight: tp.Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Invokes the `BinaryCrossentropy` instance.

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

        return binary_crossentropy(
            y_true,
            y_pred,
            from_logits=self._from_logits,
            label_smoothing=self._label_smoothing,
        )
