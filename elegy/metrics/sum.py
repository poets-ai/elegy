from elegy.metrics import reduce


class Sum(reduce.Reduce):
    """Computes the (weighted) sum of the given values.
  For example, if values is [1, 3, 5, 7] then the sum is 16.
  If the weights were specified as [1, 1, 0, 0] then the sum would be 4.
  This metric creates one variable, `total`, that is used to compute the sum of
  `values`. This is ultimately returned as `sum`.
  If `sample_weight` is `None`, weights default to 1.  Use `sample_weight` of 0
  to mask values.
  Usage:
  >>> m = tf.keras.metrics.Sum()
  >>> _ = m.update_state([1, 3, 5, 7])
  >>> m.result().numpy()
  16.0
  Usage with tf.keras API:
  ```python
  model = tf.keras.Model(inputs, outputs)
  model.add_metric(tf.keras.metrics.Sum(name='sum_1')(outputs))
  model.compile('sgd', loss='mse')
  ```
  """

    def __init__(self, name=None, dtype=None):
        """Creates a `Sum` instance.
    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    """
        super(Sum, self).__init__(
            reduction=reduce.Reduction.SUM, name=name, dtype=dtype
        )

