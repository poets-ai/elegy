from elegy.initializers import TruncatedNormal
from elegy.types import Initializer
from elegy import module, hooks
import typing as tp
import jax.numpy as jnp
import haiku as hk

import numpy as np


class Linear(module.Module):
    """Linear module."""

    w: np.ndarray
    b: np.ndarray

    def __init__(
        self,
        output_size: int,
        with_bias: bool = True,
        w_init: tp.Optional[Initializer] = None,
        b_init: tp.Optional[Initializer] = None,
        **kwargs
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
            kwargs: Additional keyword arguments passed to Module.
        """
        super().__init__(**kwargs)
        self.input_size = None
        self.output_size = output_size
        self.with_bias = with_bias
        self.w_init = w_init
        self.b_init = b_init or jnp.zeros

    def call(self, inputs: np.ndarray) -> np.ndarray:
        """
        """
        if not inputs.shape:
            raise ValueError("Input must not be scalar.")

        input_size = self.input_size = inputs.shape[-1]
        output_size = self.output_size
        dtype = inputs.dtype

        w_init = self.w_init

        if w_init is None:
            stddev = 1.0 / np.sqrt(self.input_size)
            w_init = TruncatedNormal(stddev=stddev)

        w = hooks.get_parameter(
            "w", [input_size, output_size], dtype, initializer=w_init
        )

        out = jnp.dot(inputs, w)

        if self.with_bias:
            b = hooks.get_parameter(
                "b", [self.output_size], dtype, initializer=self.b_init
            )
            b = jnp.broadcast_to(b, out.shape)
            out = out + b

        return out
