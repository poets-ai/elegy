import typing as tp

import jax.numpy as jnp

from elegy.metrics import reduce


class Sum(reduce.Reduce):
    """
    Computes the (weighted) sum of the given values.
    
    For example, if values is [1, 3, 5, 7] then the sum is 16.
    If the weights were specified as [1, 1, 0, 0] then the sum would be 4.
    This metric creates one variable, `total`, that is used to compute the sum of
    `values`. This is ultimately returned as `sum`.
    If `sample_weight` is `None`, weights default to 1.  Use `sample_weight` of 0
    to mask values.
    
    Usage:
    ```python
    >>> m = elegy.metrics.Sum()
    >>> _ = m.update_state([1, 3, 5, 7])
    >>> m.result().numpy()
    16.0
    ``` 
    Usage with elegy API:
    ```python
    model = elegy.Model(inputs, outputs)
    model.add_metric(elegy.metrics.Sum(name='sum_1')(outputs))
    model.compile('sgd', loss='mse')
    ```
    """

    def __init__(
        self, name: tp.Optional[str] = None, dtype: tp.Optional[jnp.dtype] = None,
    ):
        """Creates a `Sum` instance.

        Arguments:
          name: (Optional) string name of the metric instance.
          dtype: (Optional) data type of the metric result.
        """
        super().__init__(reduction=reduce.Reduction.SUM, name=name, dtype=dtype)
