import typing as tp

import jax.numpy as jnp

from elegy.losses.mean_squared_error import mean_squared_error
from elegy.metrics.mean_metric_wrapper import MeanMetricWrapper


class MeanSquaredError(MeanMetricWrapper):
    """Computes the mean squared error between `y_true` and `y_pred`.
  Usage:
  >>> m = elegy.metrics.MeanSquaredError()
  >>> _ = m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
  >>> m.result().numpy()
  0.25
  >>> m.reset_states()
  >>> _ = m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
  ...                    sample_weight=[1, 0])
  >>> m.result().numpy()
  0.5
  Usage with elegy API:
  ```python
  model = elegy.Model(inputs, outputs)
  model.compile(
      'sgd', loss='mse', metrics=[elegy.metrics.MeanSquaredError()])
  ```
  """

    def __init__(self, **kwargs):
        super().__init__(mean_squared_error, **kwargs)

