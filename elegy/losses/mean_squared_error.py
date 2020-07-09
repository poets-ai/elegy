from elegy import utils
from elegy.losses.loss import Loss, Reduction
import jax.numpy as jnp


def mean_squared_error(y_true: jnp.ndarray, y_pred: jnp.ndarray):
    """Computes the mean squared error between labels and predictions.
    After computing the squared distance between the inputs, the mean value over
    the last dimension is returned.
    `loss = mean(square(y_true - y_pred), axis=-1)`
    Usage:
    >>> y_true = np.random.randint(0, 2, size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = elegy.losses.mean_squared_error(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> assert np.array_equal(
    ...     loss.numpy(), np.mean(np.square(y_true - y_pred), axis=-1))
    Arguments:
      y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
      y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    Returns:
      Mean squared error values. shape = `[batch_size, d0, .. dN-1]`.
    """

    y_true = y_true.astype(y_pred.dtype)

    return jnp.mean(jnp.square(y_pred - y_true), axis=-1)


class MeanSquaredError(Loss):
    """Computes the mean of squares of errors between labels and predictions.
    `loss = square(y_true - y_pred)`
    Usage:
    >>> y_true = [[0., 1.], [0., 0.]]
    >>> y_pred = [[1., 1.], [1., 0.]]
    >>> # Using 'auto'/'sum_over_batch_size' reduction type.
    >>> mse = elegy.losses.MeanSquaredError()
    >>> mse(y_true, y_pred).numpy()
    0.5
    >>> # Calling with 'sample_weight'.
    >>> mse(y_true, y_pred, sample_weight=[0.7, 0.3]).numpy()
    0.25
    >>> # Using 'sum' reduction type.
    >>> mse = elegy.losses.MeanSquaredError(
    ...     reduction=elegy.losses.Reduction.SUM)
    >>> mse(y_true, y_pred).numpy()
    1.0
    >>> # Using 'none' reduction type.
    >>> mse = elegy.losses.MeanSquaredError(
    ...     reduction=elegy.losses.Reduction.NONE)
    >>> mse(y_true, y_pred).numpy()
    array([0.5, 0.5], dtype=float32)
    Usage with the `compile` API:
    ```python
    model = elegy.Model(inputs, outputs)
    model.compile('sgd', loss=elegy.losses.MeanSquaredError())
    ```
    """

    @utils.inject_dependencies
    def call(self, y_true: jnp.ndarray, y_pred: jnp.ndarray):
        return mean_squared_error(y_true, y_pred)
