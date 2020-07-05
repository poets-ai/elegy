import typing as tp

import jax.numpy as jnp

from elegy.metrics.mean_metric_wrapper import MeanMetricWrapper


def accuracy(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    # [y_pred, y_true], _ = metrics_utils.ragged_assert_compatible_and_get_flat_values(
    #     [y_pred, y_true]
    # )
    # y_pred.shape.assert_is_compatible_with(y_true.shape)

    if y_true.dtype != y_pred.dtype:
        y_pred = y_pred.astype(y_true.dtype)

    if len(y_pred.shape) > len(y_true.shape):
        y_pred = jnp.argmax(y_pred, axis=-1)

    return (y_true == y_pred).astype(jnp.float32)


class Accuracy(MeanMetricWrapper):
    """Calculates how often predictions equals labels.
  This metric creates two local variables, `total` and `count` that are used to
  compute the frequency with which `y_pred` matches `y_true`. This frequency is
  ultimately returned as `binary accuracy`: an idempotent operation that simply
  divides `total` by `count`.
  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.
  Usage:
  >>> m = tf.keras.metrics.Accuracy()
  >>> _ = m.update_state([[1], [2], [3], [4]], [[0], [2], [3], [4]])
  >>> m.result().numpy()
  0.75
  >>> m.reset_states()
  >>> _ = m.update_state([[1], [2], [3], [4]], [[0], [2], [3], [4]],
  ...                    sample_weight=[1, 1, 0, 0])
  >>> m.result().numpy()
  0.5
  Usage with tf.keras API:
  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile('sgd', loss='mse', metrics=[tf.keras.metrics.Accuracy()])
  ```
  """

    def __init__(
        self, name: tp.Optional[str] = None, dtype: tp.Optional[jnp.dtype] = None
    ):
        super().__init__(accuracy, name, dtype=dtype)

