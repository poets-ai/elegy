import typing as tp

import jax.numpy as jnp

from elegy.losses.mean_squared_error import mean_squared_error
from elegy.metrics.mean_metric_wrapper import MeanMetricWrapper


class MeanSquaredError(MeanMetricWrapper):
    """Computes the mean squared error between `y_true` and `y_pred`.
  Usage:
  >>> m = tf.keras.metrics.MeanSquaredError()
  >>> _ = m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
  >>> m.result().numpy()
  0.25
  >>> m.reset_states()
  >>> _ = m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
  ...                    sample_weight=[1, 0])
  >>> m.result().numpy()
  0.5
  Usage with tf.keras API:
  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile(
      'sgd', loss='mse', metrics=[tf.keras.metrics.MeanSquaredError()])
  ```
  """

    def __init__(
        self, name: tp.Optional[str] = None, dtype: tp.Optional[jnp.dtype] = None
    ):
        super().__init__(mean_squared_error, name=name, dtype=dtype)

