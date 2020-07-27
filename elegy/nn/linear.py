from elegy import hooks
from elegy.utils import Deferable
import typing as tp

import haiku as hk
import numpy as np


class Linear(hk.Linear, Deferable):
    """Linear module."""

    def __init__(
        self,
        output_size: int,
        with_bias: bool = True,
        w_init: tp.Optional[hk.initializers.Initializer] = None,
        b_init: tp.Optional[hk.initializers.Initializer] = None,
        name: tp.Optional[str] = None,
    ):
        """
        Constructs the Linear module.

        Arguments:
            output_size: Output dimensionality.
            with_bias: Whether to add a bias to the output.
            w_init: Optional initializer for weights. By default, uses random values
                from truncated normal, with stddev `1 / sqrt(fan_in)`. See
                https://arxiv.org/abs/1502.03167v3.
            b_init: Optional initializer for bias. By default, zero.
            name: Name of the module.
        """
        super().__init__(
            output_size=output_size,
            with_bias=with_bias,
            w_init=w_init,
            b_init=b_init,
            name=name,
        )

    def __call__(self, inputs: np.ndarray):
        """
        Arguments:
            inputs: Input array.
        """
        outputs = super().__call__(inputs)

        hooks.add_summary(None, self.__class__.__name__, outputs)

        return outputs
