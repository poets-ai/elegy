from elegy import types
import typing as tp
import jax.numpy as jnp
from elegy import utils
from elegy.losses.loss import Loss, Reduction


def mean_percentage_absolute_error(
    y_true: jnp.ndarray, y_pred: jnp.ndarray
) -> jnp.ndarray:
    """
    Computes the mean absolute percentage error (MAPE) between labels and predictions.
    
    After computing the absolute distance between the true value and the prediction value
    and divide by the true value, the mean value over the last dimension is returned. 
    
    Usage:
    
    ```python
    rng = jax.random.PRNGKey(42)

    y_true = jax.random.randint(rng, shape=(2, 3), minval=0, maxval=2)
    y_pred = jax.random.uniform(rng, shape=(2, 3))

    loss = elegy.losses.mean_percentage_absolute_error(y_true, y_pred)

    assert loss.shape == (2,)

    assert jnp.array_equal(loss, 100. * jnp.mean(jnp.abs((y_pred - y_true) / jnp.clip(y_true, utils.EPSILON, None))))
    ```
    
    Arguments:
        y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

    Returns:
        Mean absolute percentage error values. shape = `[batch_size, d0, .. dN-1]`.
    """
    y_true = y_true.astype(y_pred.dtype)
    diff = jnp.abs((y_pred - y_true) / jnp.maximum(jnp.abs(y_true), utils.EPSILON))
    return 100.0 * jnp.mean(diff, axis=-1)


class MeanAbsolutePercentageError(Loss):
    """
    Computes the mean absolute errors between labels and predictions.

    `loss = mean(abs((y_true - y_pred) / y_true))`

    Usage:

    ```python
    y_true = jnp.array([[1.0, 1.0], [0.9, 0.0]])
    y_pred = jnp.array([[1.0, 1.0], [1.0, 0.0]])

    # Using 'auto'/'sum_over_batch_size' reduction type.
    mape = elegy.losses.MeanAbsolutePercentageError()
    result = mape(y_true, y_pred)
    assert jnp.isclose(result, 2.78, rtol=0.01)

    # Calling with 'sample_weight'.
    assert jnp.isclose(mape(y_true, y_pred, sample_weight=jnp.array([0.1, 0.9])), 2.5, rtol=0.01)

    # Using 'sum' reduction type.
    mape = elegy.losses.MeanAbsolutePercentageError(reduction=elegy.losses.Reduction.SUM)

    assert jnp.isclose(mape(y_true, y_pred), 5.6, rtol=0.01)

    # Using 'none' reduction type.
    mape = elegy.losses.MeanAbsolutePercentageError(reduction=elegy.losses.Reduction.NONE)

    assert jnp.all(jnp.isclose(result, [0. , 5.6], rtol=0.01))

    ```
    Usage with the Elegy API:

    ```python
    model = elegy.Model(
        module_fn,
        loss=elegy.losses.MeanAbsolutePercentageError(),
        metrics=elegy.metrics.Mean.defer(),
    )
    ```
    """

    def __init__(
        self,
        reduction: tp.Optional[Reduction] = None,
        name: tp.Optional[str] = None,
        weight: tp.Optional[float] = None,
        on: tp.Optional[types.IndexLike] = None,
    ):
        """
        Initializes `Mean` class.

        Arguments:
            reduction: (Optional) Type of `elegy.losses.Reduction` to apply to
                loss. Default value is `SUM_OVER_BATCH_SIZE`. For almost all cases
                this defaults to `SUM_OVER_BATCH_SIZE`. When used with
                `tf.distribute.Strategy`, outside of built-in training loops such as
                `elegy` `compile` and `fit`, or `SUM_OVER_BATCH_SIZE`
                will raise an error.
                for more details.
            name: Optional name for the loss.
            weight: Optional weight contribution for the total loss. Defaults to `1`.
            on: A string or integer, or iterable of string or integers, that
                indicate how to index/filter the `y_true` and `y_pred`
                arguments before passing them to `call`. For example if `on = "a"` then
                `y_true = y_true["a"]`. If `on` is an iterable
                the structures will be indexed iteratively, for example if `on = ["a", 0, "b"]`
                then `y_true = y_true["a"][0]["b"]`, same for `y_pred`. For more information
                check out [Keras-like behavior](https://poets-ai.github.io/elegy/guides/modules-losses-metrics/#keras-like-behavior).
        """

        return super().__init__(reduction=reduction, name=name, weight=weight, on=on)

    def __apply__(
        self,
        y_true: jnp.ndarray,
        y_pred: jnp.ndarray,
        sample_weight: tp.Optional[
            jnp.ndarray
        ] = None,  # not used, __call__ handles it, left for documentation purposes.
    ) -> jnp.ndarray:
        """
        Invokes the `MeanAbsolutePercentageError` instance.

        Arguments:
            y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`, except
                sparse loss functions such as sparse categorical crossentropy where
                shape = `[batch_size, d0, .. dN-1]`
            y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`
            sample_weight: Optional `sample_weight` acts as a
                coefficient for the loss. If a scalar is provided, then the loss is
                simply scaled by the given value. If `sample_weight` is a tensor of size
                `[batch_size]`, then the total loss for each sample of the batch is
                rescaled by the corresponding element in the `sample_weight` vector. If
                the shape of `sample_weight` is `[batch_size, d0, .. dN-1]` (or can be
                broadcasted to this shape), then each loss element of `y_pred` is scaled
                by the corresponding value of `sample_weight`. (Note on`dN-1`: all loss
                functions reduce by 1 dimension, usually axis=-1.)

        Returns:
            Weighted loss float `Tensor`. If `reduction` is `NONE`, this has
                shape `[batch_size, d0, .. dN-1]`; otherwise, it is scalar. (Note `dN-1`
                because all loss functions reduce by 1 dimension, usually axis=-1.)
        
        Raises:
            ValueError: If the shape of `sample_weight` is invalid.
        """
        return mean_percentage_absolute_error(y_true, y_pred)
