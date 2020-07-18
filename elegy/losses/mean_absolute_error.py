from elegy import types
import typing as tp

import jax.numpy as jnp

from elegy import utils
from elegy.losses.loss import Loss, Reduction


def mean_absolute_error(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the mean absolute error between labels and predictions.
    
    After computing the absolute distance between the inputs, the mean value over
    the last dimension is returned.
    
    ```python
    loss = mean(abs(y_true - y_pred), axis=-1)
    ```

    Usage:
    
    ```python
    rng = jax.random.PRNGKey(42)

    y_true = jax.random.randint(rng, shape=(2, 3), minval=0, maxval=2)
    y_pred = jax.random.uniform(rng, shape=(2, 3))

    loss = elegy.losses.mean_absolute_error(y_true, y_pred)

    assert loss.shape == (2,)

    assert jnp.array_equal(loss, jnp.mean(jnp.abs(y_true - y_pred), axis=-1))
    ```
    
    Arguments:
        y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    
    Returns:
        Mean absolute error values. shape = `[batch_size, d0, .. dN-1]`.
    """

    y_true = y_true.astype(y_pred.dtype)

    return jnp.mean(jnp.abs(y_pred - y_true), axis=-1)


class MeanAbsoluteError(Loss):
    """
    Computes the mean absolute errors between labels and predictions.

    `loss = mean(abs(y_true - y_pred))`

    Usage:

    ```python
    y_true = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    y_pred = jnp.array([[1.0, 1.0], [1.0, 0.0]])

    # Using 'auto'/'sum_over_batch_size' reduction type.
    mae = elegy.losses.MeanAbsoluteError()

    assert mae(y_true, y_pred) == 0.5

    # Calling with 'sample_weight'.
    assert mae(y_true, y_pred, sample_weight=jnp.array([0.7, 0.3])) == 0.25

    # Using 'sum' reduction type.
    mae = elegy.losses.MeanAbsoluteError(reduction=elegy.losses.Reduction.SUM)

    assert mae(y_true, y_pred) == 1.0

    # Using 'none' reduction type.
    mae = elegy.losses.MeanAbsoluteError(reduction=elegy.losses.Reduction.NONE)

    assert list(mae(y_true, y_pred)) == [0.5, 0.5]
    ```
    Usage with the Elegy API:

    ```python
    model = elegy.Model(
        module_fn,
        loss=elegy.losses.MeanAbsoluteError(),
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
        """

        return super().__init__(reduction=reduction, name=name, weight=weight, on=on)

    def call(
        self,
        y_true: jnp.ndarray,
        y_pred: jnp.ndarray,
        sample_weight: tp.Optional[
            jnp.ndarray
        ] = None,  # not used, __call__ handles it, left for documentation purposes.
    ) -> jnp.ndarray:
        """
        Invokes the `MeanAbsoluteError` instance.

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
        return mean_absolute_error(y_true, y_pred)
