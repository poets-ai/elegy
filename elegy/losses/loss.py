from abc import abstractmethod

from numpy.lib.arraysetops import isin
from elegy import types
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
        on: tp.Optional[types.IndexLike] = None,
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
        self._labels_filter = (on,) if isinstance(on, (str, int)) else on
        self.call = utils.inject_dependencies(self.call)

    def __call__(
        self,
        y_true=None,
        y_pred=None,
        sample_weight: tp.Optional[jnp.ndarray] = None,
        **kwargs,
    ):

        if self._labels_filter is not None:
            if y_true is not None:
                for index in self._labels_filter:
                    y_true = y_true[index]

            if y_pred is not None:
                for index in self._labels_filter:
                    y_pred = y_pred[index]

        values = self.call(
            y_true=y_true, y_pred=y_pred, sample_weight=sample_weight, **kwargs
        )

        if isinstance(values, tp.Dict):
            return {
                key: reduce_loss(values, sample_weight, self.weight, self._reduction)
                for key, values in values.items()
            }
        else:
            return reduce_loss(values, sample_weight, self.weight, self._reduction)

    @abstractmethod
    def call(self, *args, **kwargs):
        ...


def reduce_loss(values, sample_weight, weight, reduction):

    values = jnp.asarray(values)

    if sample_weight is not None:
        values *= sample_weight

    if reduction == Reduction.NONE:
        loss = values
    elif reduction == Reduction.SUM:
        loss = values.sum()
    elif reduction == Reduction.SUM_OVER_BATCH_SIZE:
        loss = values.sum() / jnp.prod(values.shape)
    else:
        raise ValueError(f"Invalid reduction '{reduction}'")

    return loss * weight
