from elegy import types
from elegy import utils
import typing as tp

import jax.numpy as jnp

from elegy.metrics import reduce


class Mean(reduce.Reduce):
    """
    Computes the (weighted) mean of the given values.
    
    For example, if values is `[1, 3, 5, 7]` then the mean is `4`.
    If the weights were specified as `[1, 1, 0, 0]` then the mean would be `2`.
    This metric creates two variables, `total` and `count` that are used to
    compute the average of `values`. This average is ultimately returned as `mean`
    which is an idempotent operation that simply divides `total` by `count`.
    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.
  
    Usage:

    ```python
    mean = elegy.metrics.Mean()
    result = mean([1, 3, 5, 7])  # 16 / 4
    assert result == 4.0


    result = mean([4, 10])  # 30 / 6
    assert result == 5.0
    ```

    Usage with elegy API:
    
    ```python
    model = elegy.Model(
        module_fn,
        loss=elegy.losses.MeanSquaredError(),
        metrics=elegy.metrics.Mean.defer(),
    )
    ```
    """

    def __init__(
        self,
        name: tp.Optional[str] = None,
        dtype: tp.Optional[jnp.dtype] = None,
        on: tp.Optional[types.IndexLike] = None,
    ):
        """Creates a `Mean` instance.
        Arguments:
          name: (Optional) string name of the metric instance.
          dtype: (Optional) data type of the metric result.
          on: A string or integer, or iterable of string or integers, that
                indicate how to index/filter the `y_true` and `y_pred`
                arguments before passing them to `call`. For example if `on = "a"` then
                `y_true = y_true["a"]`. If `on` is an iterable
                the structures will be indexed iteratively, for example if `on = ["a", 0, "b"]`
                then `y_true = y_true["a"][0]["b"]`, same for `y_pred`. For more information
                check out [Keras-like behavior](https://poets-ai.github.io/elegy/guides/modules-losses-metrics/#keras-like-behavior).
        """
        super().__init__(
            reduction=reduce.Reduction.WEIGHTED_MEAN, name=name, dtype=dtype, on=on
        )

    def call(
        self, values: jnp.ndarray, sample_weight: tp.Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Accumulates the mean statistic over various batches.
        
        Arguments:
            values: Per-example value.
            sample_weight: Optional weighting of each example.
        
        Returns:
            Array with the cumulative mean.
        """
        return super().call(values=values, sample_weight=sample_weight)
