# Implementation based on Tensorflow Keras:
# https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/losses.py#L44-L201

from abc import abstractmethod
from elegy import module
import functools
import numpy as np

from numpy.lib.arraysetops import isin
from elegy import types
from enum import Enum
import re
import typing as tp


import jax.numpy as jnp

from elegy import utils


class Reduction(Enum):
    """
    Types of loss reduction.
    
    Contains the following values:
    * `NONE`: Weighted losses with one dimension reduced (axis=-1, or axis
        specified by loss function). When this reduction type used with built-in
        Keras training loops like `fit`/`evaluate`, the unreduced vector loss is
        passed to the optimizer but the reported loss will be a scalar value.
    * `SUM`: Scalar sum of weighted losses.
    * `SUM_OVER_BATCH_SIZE`: Scalar `SUM` divided by number of elements in losses.
    """

    # AUTO = "auto"
    NONE = "none"
    SUM = "sum"
    SUM_OVER_BATCH_SIZE = "sum_over_batch_size"

    @classmethod
    def all(cls):
        return (
            # cls.AUTO,
            cls.NONE,
            cls.SUM,
            cls.SUM_OVER_BATCH_SIZE,
        )

    @classmethod
    def validate(cls, key):
        if key not in cls.all():
            raise ValueError("Invalid Reduction Key %s." % key)


class Loss(module.Module):
    """
    Loss base class.

    To be implemented by subclasses:

    * `call()`: Contains the logic for loss calculation.

    Example subclass implementation:

    ```python
    class MeanSquaredError(Loss):
        def call(self, y_true, y_pred):
            return jnp.mean(jnp.square(y_pred - y_true), axis=-1)
    ```

    Please see the [Modules, Losses, and Metrics Guide]
    (https://poets-ai.github.io/elegy/guides/modules-losses-metrics/#losses) for more
    details on this.
    """

    def __init__(
        self,
        reduction: tp.Optional[Reduction] = None,
        weight: tp.Optional[float] = None,
        on: tp.Optional[types.IndexLike] = None,
        **kwargs,
    ):
        """
        Initializes `Loss` class.
        
        Arguments:
            reduction: (Optional) Type of `elegy.losses.Reduction` to apply to
                loss. Default value is `SUM_OVER_BATCH_SIZE`. For almost all cases
                this defaults to `SUM_OVER_BATCH_SIZE`.
            weight: Optional weight contribution for the total loss. Defaults to `1`.
            on: A string or integer, or iterable of string or integers, that
                indicate how to index/filter the `y_true` and `y_pred`
                arguments before passing them to `call`. For example if `on = "a"` then
                `y_true = y_true["a"]`. If `on` is an iterable
                the structures will be indexed iteratively, for example if `on = ["a", 0, "b"]`
                then `y_true = y_true["a"][0]["b"]`, same for `y_pred`. For more information
                check out [Keras-like behavior](https://poets-ai.github.io/elegy/guides/modules-losses-metrics/#keras-like-behavior).
        """
        super().__init__(**kwargs)
        self.weight = weight if weight is not None else 1.0
        self._reduction = (
            reduction if reduction is not None else Reduction.SUM_OVER_BATCH_SIZE
        )
        self._labels_filter = (on,) if isinstance(on, (str, int)) else on

    def __call__(
        self, *args, **kwargs,
    ):

        if self._labels_filter is not None:
            if "y_true" in kwargs and kwargs["y_true"] is not None:
                for index in self._labels_filter:
                    kwargs["y_true"] = kwargs["y_true"][index]

            if "y_pred" in kwargs and kwargs["y_pred"] is not None:
                for index in self._labels_filter:
                    kwargs["y_pred"] = kwargs["y_pred"][index]

        values = super().__call__(*args, **kwargs)

        sample_weight: tp.Optional[jnp.ndarray] = kwargs.get("sample_weight", None)

        if isinstance(values, tp.Dict):
            return {
                key: reduce_loss(values, sample_weight, self.weight, self._reduction)
                for key, values in values.items()
            }
        else:
            return reduce_loss(values, sample_weight, self.weight, self._reduction)

    @abstractmethod
    def call(self, *args, **kwargs) -> tp.Any:
        ...


def reduce_loss(values, sample_weight, weight, reduction):

    values: jnp.ndarray = jnp.asarray(values)

    if sample_weight is not None:
        values *= sample_weight

    if reduction == Reduction.NONE:
        loss = values
    elif reduction == Reduction.SUM:
        loss = jnp.sum(values)
    elif reduction == Reduction.SUM_OVER_BATCH_SIZE:
        loss = jnp.sum(values) / jnp.prod(values.shape)
    else:
        raise ValueError(f"Invalid reduction '{reduction}'")

    return loss * weight
