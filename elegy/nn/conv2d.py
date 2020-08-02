from haiku._src.typing import PadFnOrFns
from elegy import hooks
import typing as tp

import haiku as hk
import numpy as np


class Conv2D(hk.Conv2D):
    """Two dimensional convolution."""

    def __init__(
        self,
        output_channels: int,
        kernel_shape: tp.Union[int, tp.Sequence[int]],
        stride: tp.Union[int, tp.Sequence[int]] = 1,
        rate: tp.Union[int, tp.Sequence[int]] = 1,
        padding: tp.Union[str, tp.Sequence[tp.Tuple[int, int]], PadFnOrFns] = "SAME",
        with_bias: bool = True,
        w_init: tp.Optional[hk.initializers.Initializer] = None,
        b_init: tp.Optional[hk.initializers.Initializer] = None,
        data_format: str = "NHWC",
        mask: tp.Optional[np.ndarray] = None,
        name: tp.Optional[str] = None,
    ):
        """
        Initializes the module.

        Arguments:
            output_channels: Number of output channels.
            kernel_shape: The shape of the kernel. Either an integer or a sequence of
                length 2.
            stride: tp.Optional stride for the kernel. Either an integer or a sequence of
                length 2. Defaults to 1.
            rate: tp.Optional kernel dilation rate. Either an integer or a sequence of
                length 2. 1 corresponds to standard ND convolution,
                ``rate > 1`` corresponds to dilated convolution. Defaults to 1.
            padding: tp.Optional padding algorithm. Either ``VALID`` or ``SAME`` or
                a callable or sequence of callables of length 2. Any callables must
                take a single integer argument equal to the effective kernel size and
                return a list of two integers representing the padding before and after.
                See haiku.pad.* for more details and example functions.
                Defaults to ``SAME``. See:
                https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
            with_bias: Whether to add a bias. By default, true.
            w_init: tp.Optional weight initialization. By default, truncated normal.
            b_init: tp.Optional bias initialization. By default, zeros.
            data_format: The data format of the input. Either ``NHWC`` or ``NCHW``. By
                default, ``NHWC``.
            mask: tp.Optional mask of the weights.
            name: The name of the module.
        """
        super().__init__(
            output_channels=output_channels,
            kernel_shape=kernel_shape,
            stride=stride,
            rate=rate,
            padding=padding,
            with_bias=with_bias,
            w_init=w_init,
            b_init=b_init,
            data_format=data_format,
            mask=mask,
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
