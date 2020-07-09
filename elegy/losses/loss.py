from abc import abstractmethod
from enum import Enum
import re
import typing as tp

import haiku as hk
import jax.numpy as jnp


class Reduction(Enum):
    """Types of loss reduction.
    Contains the following values:
    * `AUTO`: Indicates that the reduction option will be determined by the usage
        context. For almost all cases this defaults to `SUM_OVER_BATCH_SIZE`. When
        used with `tf.distribute.Strategy`, outside of built-in training loops such
        as `elegy` `compile` and `fit`, we expect reduction value to be
        `SUM` or `NONE`. Using `AUTO` in that case will raise an error.
    * `NONE`: Weighted losses with one dimension reduced (axis=-1, or axis
        specified by loss function). When this reduction type used with built-in
        Keras training loops like `fit`/`evaluate`, the unreduced vector loss is
        passed to the optimizer but the reported loss will be a scalar value.
    * `SUM`: Scalar sum of weighted losses.
    * `SUM_OVER_BATCH_SIZE`: Scalar `SUM` divided by number of elements in losses.
        This reduction type is not supported when used with
        `tf.distribute.Strategy` outside of built-in training loops like `elegy`
        `compile`/`fit`.
        You can implement 'SUM_OVER_BATCH_SIZE' using global batch size like:
        ```
        with strategy.scope():
        loss_obj = elegy.losses.CategoricalCrossentropy(
            reduction=elegy.losses.Reduction.NONE)
        ....
        loss = tf.reduce_sum(loss_object(labels, predictions)) *
            (1. / global_batch_size)
        ```
    Please see the
    [custom training guide](https://www.tensorflow.org/tutorials/distribute/custom_training)  # pylint: disable=line-too-long
    for more details on this.
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


class Loss:
    """Wraps a loss function in the `Loss` class."""

    def __init__(
        self,
        reduction: tp.Optional[Reduction] = None,
        name: tp.Optional[str] = None,
        weight: float = 1.0,
    ):
        """Initializes `Loss` class.
        Arguments:
        fn: The loss function to wrap, with signature `fn(y_true, y_pred,
            **kwargs)`.
        reduction: (Optional) Type of `elegy.losses.Reduction` to apply to
            loss. Default value is `AUTO`. `AUTO` indicates that the reduction
            option will be determined by the usage context. For almost all cases
            this defaults to `SUM_OVER_BATCH_SIZE`. When used with
            `tf.distribute.Strategy`, outside of built-in training loops such as
            `elegy` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
            will raise an error. Please see this custom training [tutorial]
            (https://www.tensorflow.org/tutorials/distribute/custom_training)
            for more details.
        name: (Optional) name for the loss.
        **kwargs: The keyword arguments that are passed on to `fn`.
        """
        self.name = (
            name
            if name is not None
            else re.sub(r"(?<!^)(?=[A-Z])", "_", self.__class__.__name__).lower()
        )
        self.weight = weight
        self._reduction = (
            reduction if reduction is not None else Reduction.SUM_OVER_BATCH_SIZE
        )

    def __call__(self, *args, sample_weight: tp.Optional[jnp.ndarray] = None, **kwargs):
        values = self.call(*args, sample_weight=sample_weight, **kwargs)

        if sample_weight is not None:
            values *= sample_weight

        if self._reduction == Reduction.NONE:
            loss = values
        elif self._reduction == Reduction.SUM:
            loss = values.sum()
        elif self._reduction == Reduction.SUM_OVER_BATCH_SIZE:
            loss = values.sum() / jnp.prod(values.shape)
        else:
            raise ValueError(f"Invalid reduction '{self._reduction}'")

        return loss * self.weight

    @abstractmethod
    def call(self, *args, **kwargs):
        ...

