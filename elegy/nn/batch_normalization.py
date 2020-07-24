# Implementation based on Tensorflow Keras and Haiku
# Tensorflow: https://github.com/tensorflow/tensorflow/blob/2b96f3662bd776e277f86997659e61046b56c315/tensorflow/python/keras/layers/normalization.py#L46
# Haiku: https://github.com/deepmind/dm-haiku/blob/master/haiku/_src/batch_norm.py#L39#L194

import typing as tp

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from haiku._src import utils as hk_utils

from elegy.module import Module


class BatchNormalization(Module):
    r"""
    Normalize and create_scale inputs or activations. (Ioffe and Szegedy, 2014).

    Normalize the activations of the previous layer at each batch,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    
    ### Normalization equations

    Consider the intermediate activations \(x\) of a mini-batch of size
    \\(m\\):
    We can compute the mean and variance of the batch
    
    \\({\mu_B} = \frac{1}{m} \sum_{i=1}^{m} {x_i}\\)
    \\({\sigma_B^2} = \frac{1}{m} \sum_{i=1}^{m} ({x_i} - {\mu_B})^2\\)
    
    and then compute a normalized \\(x\\), including a small factor
    \\({\eps}\\) for numerical stability.
    
    \\(\hat{x_i} = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \eps}}\\)
    
    And finally \\(\hat{x}\\) is linearly transformed by \\({\gamma}\\)
    and \\({\beta}\\), which are learned parameters:
    
    \\({y_i} = {\gamma * \hat{x_i} + \beta}\\)
    
    ### References

    * [Batch Normalization: Accelerating Deep Network Training by Reducing
        Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
    """

    def __init__(
        self,
        axis: tp.Union[int, tp.Sequence[int], None] = None,
        momentum: float = 0.99,
        eps: float = 1e-3,
        create_offset: bool = True,
        create_scale: bool = True,
        scale_init: tp.Optional[hk.initializers.Initializer] = None,
        offset_init: tp.Optional[hk.initializers.Initializer] = None,
        cross_replica_axis: tp.Optional[str] = None,
        data_format: str = "channels_last",
        name: tp.Optional[str] = None,
    ):
        """
        Creates a BatchNormalization instance.
        
        Arguments:
            axis: Integer, the axis that should be normalized
                (typically the features axis).
                For instance, after a `Conv2D` layer with
                `data_format="channels_first"`,
                set `axis=1` in `BatchNormalization`.
            momentum: Momentum for the moving average.
            eps: Small float added to variance to avoid dividing by zero.
            create_offset: If True, add offset of `beta` to normalized tensor.
                se, `beta` is ignored.
            create_scale: If True, multiply by `gamma`.
                If False, `gamma` is not used.
                When the next layer is linear (also e.g. `relu`),
                this can be disabled since the scaling
                will be done by the next layer.
            scale_init: Optional initializer for gain (aka create_scale). Can only be set
                if ``create_scale=True``. By default, ``1``.
            offset_init: Optional initializer for bias (aka offset). Can only be set
                if ``create_offset=True``. By default, ``0``.
            axis: Which axes to reduce over. The default (``None``) signifies that all
                but the channel axis should be normalized. Otherwise this is a list of
                axis indices which will have normalization statistics calculated.
            cross_replica_axis: If not ``None``, it should be a string representing
                the axis name over which this module is being run within a ``jax.pmap``.
                Supplying this argument means that batch statistics are calculated
                across all replicas on that axis.
            data_format: The data format of the input. Can be either
                ``channels_first``, ``channels_last``, ``N...C`` or ``NC...``. By
                default it is ``channels_last``.
            name: The module name.
        """
        super().__init__(name=name)

        if not create_scale and scale_init is not None:
            raise ValueError("Cannot set `scale_init` if `create_scale=False`")
        if not create_offset and offset_init is not None:
            raise ValueError("Cannot set `offset_init` if `create_offset=False`")

        self.create_scale = create_scale
        self.create_offset = create_offset
        self.eps = eps
        self.scale_init = scale_init or jnp.ones
        self.offset_init = offset_init or jnp.zeros
        self.axis = axis
        self.cross_replica_axis = cross_replica_axis
        self.channel_index = hk_utils.get_channel_index(data_format)
        self.mean_ema = hk.ExponentialMovingAverage(momentum, name="mean_ema")
        self.var_ema = hk.ExponentialMovingAverage(momentum, name="var_ema")

        hk.BatchNorm

    def call(
        self,
        inputs: jnp.ndarray,
        is_training: bool,
        test_local_stats: bool = False,
        create_scale: tp.Optional[jnp.ndarray] = None,
        offset: tp.Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Computes the normalized version of the input.

    Arguments:
        inputs: An array, where the data format is ``[..., C]``.
        is_training: Whether training is currently happening.
        test_local_stats: Whether local stats are used when is_training=False.
        create_scale: An array up to n-D. The shape of this tensor must be broadcastable
            to the shape of ``inputs``. This is the create_scale applied to the normalized
            inputs. This cannot be passed in if the module was constructed with
            ``create_scale=True``.
        offset: An array up to n-D. The shape of this tensor must be broadcastable
            to the shape of ``inputs``. This is the offset applied to the normalized
            inputs. This cannot be passed in if the module was constructed with
            ``create_offset=True``.

    Returns:
        The array, normalized across all but the last dimension.
    """
        if self.create_scale and create_scale is not None:
            raise ValueError(
                "Cannot pass `create_scale` at call time if `create_scale=True`."
            )
        if self.create_offset and offset is not None:
            raise ValueError(
                "Cannot pass `offset` at call time if `create_offset=True`."
            )

        channel_index = self.channel_index
        if channel_index < 0:
            channel_index += inputs.ndim

        if self.axis is not None:
            axis = self.axis
        else:
            axis = [i for i in range(inputs.ndim) if i != channel_index]

        if is_training or test_local_stats:
            cross_replica_axis = self.cross_replica_axis
            if self.cross_replica_axis:
                mean = jnp.mean(inputs, axis, keepdims=True)
                mean = jax.lax.pmean(mean, cross_replica_axis)
                mean_of_squares = jnp.mean(inputs ** 2, axis, keepdims=True)
                mean_of_squares = jax.lax.pmean(mean_of_squares, cross_replica_axis)
                var = mean_of_squares - mean ** 2
            else:
                mean = jnp.mean(inputs, axis, keepdims=True)
                # This uses E[(X - E[X])^2].
                # TODO(tycai): Consider the faster, but possibly less stable
                # E[X^2] - E[X]^2 method.
                var = jnp.var(inputs, axis, keepdims=True)
        else:
            mean = self.mean_ema.average
            var = self.var_ema.average

        if is_training:
            self.mean_ema(mean)
            self.var_ema(var)

        w_shape = [1 if i in axis else inputs.shape[i] for i in range(inputs.ndim)]
        w_dtype = inputs.dtype

        if self.create_scale:
            create_scale = hk.get_parameter(
                "create_scale", w_shape, w_dtype, self.scale_init
            )
        elif create_scale is None:
            create_scale = np.ones([], dtype=w_dtype)

        if self.create_offset:
            offset = hk.get_parameter("offset", w_shape, w_dtype, self.offset_init)
        elif offset is None:
            offset = np.zeros([], dtype=w_dtype)

        inv = create_scale * jax.lax.rsqrt(var + self.eps)
        return (inputs - mean) * inv + offset
