from elegy.metrics import reduce


class Mean(reduce.Reduce):
    """Computes the (weighted) mean of the given values.
  For example, if values is [1, 3, 5, 7] then the mean is 4.
  If the weights were specified as [1, 1, 0, 0] then the mean would be 2.
  This metric creates two variables, `total` and `count` that are used to
  compute the average of `values`. This average is ultimately returned as `mean`
  which is an idempotent operation that simply divides `total` by `count`.
  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.
  Usage:
  >>> m = tf.keras.metrics.Mean()
  >>> _ = m.update_state([1, 3, 5, 7])
  >>> m.result().numpy()
  4.0
  >>> m.reset_states()
  >>> _ = m.update_state([1, 3, 5, 7], sample_weight=[1, 1, 0, 0])
  >>> m.result().numpy()
  2.0
  Usage with tf.keras API:
  ```python
  model = tf.keras.Model(inputs, outputs)
  model.add_metric(tf.keras.metrics.Mean(name='mean_1')(outputs))
  model.compile('sgd', loss='mse')
  ```
  """

    def __init__(self, **kwargs):
        """Creates a `Mean` instance.
        Args:
          name: (Optional) string name of the metric instance.
          dtype: (Optional) data type of the metric result.
        """
        super().__init__(reduction=reduce.Reduction.WEIGHTED_MEAN, **kwargs)

