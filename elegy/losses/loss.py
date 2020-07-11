from abc import abstractmethod
from enum import Enum
import re
import typing as tp


import haiku as hk
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


class Loss:
    """Wraps a loss function in the `Loss` class."""

    def __init__(
        self,
        reduction: tp.Optional[Reduction] = None,
        name: tp.Optional[str] = None,
        weight: tp.Optional[float] = None,
    ):
        """
        Initializes `Loss` class.
        
        Arguments:
            reduction: (Optional) Type of `elegy.losses.Reduction` to apply to
                loss. Default value is `SUM_OVER_BATCH_SIZE`. For almost all cases
                this defaults to `SUM_OVER_BATCH_SIZE`. When used with
                `tf.distribute.Strategy`, outside of built-in training loops such as
                `elegy` `compile` and `fit`, or `SUM_OVER_BATCH_SIZE`
                will raise an error.
                for more details.
            name: Optional name for the loss.
            weight: Optional weight contribution for the total loss. Defaults to `1`.
        """
        self.name = (
            name
            if name is not None
            else re.sub(r"(?<!^)(?=[A-Z])", "_", self.__class__.__name__).lower()
        )
        self.weight = weight if weight is not None else 1.0
        self._reduction = (
            reduction if reduction is not None else Reduction.SUM_OVER_BATCH_SIZE
        )
        self.call = utils.inject_dependencies(self.call)

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
