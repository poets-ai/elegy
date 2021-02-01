# Implementation based on Tensorflow Keras and Haiku
# Tensorflow: https://github.com/tensorflow/tensorflow/blob/2b96f3662bd776e277f86997659e61046b56c315/tensorflow/python/keras/layers/normalization.py#L46
# Haiku: https://github.com/deepmind/dm-haiku/blob/master/haiku/_src/batch_norm.py#L39#L194

import typing as tp
from typing import Optional, Sequence

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from haiku._src import utils as haiku_utils

from elegy import initializers, module, types
from elegy import module
from elegy import hooks
from elegy.nn.moving_averages import ExponentialMovingAverage


class BatchNormalization(module.Module):
    """Normalizes inputs to maintain a mean of ~0 and stddev of ~1.

    See: https://arxiv.org/abs/1502.03167.

    There are many different variations for how users want to manage scale and
    offset if they require them at all. These are:

      - No scale/offset in which case ``create_*`` should be set to ``False`` and
        ``scale``/``offset`` aren't passed when the module is called.
      - Trainable scale/offset in which case ``create_*`` should be set to
        ``True`` and again ``scale``/``offset`` aren't passed when the module is
        called. In this case this module creates and owns the ``scale``/``offset``
        variables.
      - Externally generated ``scale``/``offset``, such as for conditional
        normalization, in which case ``create_*`` should be set to ``False`` and
        then the values fed in at call time.

    NOTE: ``jax.vmap(hk.transform(BatchNorm))`` will update summary statistics and
    normalize values on a per-batch basis; we currently do *not* support
    normalizing across a batch axis introduced by vmap.
    """

    def __init__(
        self,
        create_scale: bool = True,
        create_offset: bool = True,
        decay_rate: float = 0.99,
        eps: float = 1e-5,
        scale_init: Optional[types.Initializer] = None,
        offset_init: Optional[types.Initializer] = None,
        axis: Optional[Sequence[int]] = None,
        cross_replica_axis: Optional[str] = None,
        data_format: str = "channels_last",
        **kwargs
    ):
        """Constructs a BatchNorm module.

        Args:
            create_scale: Whether to include a trainable scaling factor.
            create_offset: Whether to include a trainable offset.
            decay_rate: Decay rate for EMA.
            eps: Small epsilon to avoid division by zero variance. Defaults ``1e-5``,
                as in the paper and Sonnet.
            scale_init: Optional initializer for gain (aka scale). Can only be set
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
            kwargs: Additional keyword arguments passed to Module.
        """
        super().__init__(**kwargs)
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
        self.channel_index = haiku_utils.get_channel_index(data_format)
        self.mean_ema = ExponentialMovingAverage(decay_rate, name="mean_ema")
        self.var_ema = ExponentialMovingAverage(decay_rate, name="var_ema")

    def call(
        self,
        inputs: jnp.ndarray,
        training: tp.Optional[bool] = None,
        test_local_stats: bool = False,
        scale: Optional[jnp.ndarray] = None,
        offset: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Computes the normalized version of the input.

        Args:
            inputs: An array, where the data format is ``[..., C]``.
            training: Whether training is currently happening.
            test_local_stats: Whether local stats are used when training=False.
            scale: An array up to n-D. The shape of this tensor must be broadcastable
                to the shape of ``inputs``. This is the scale applied to the normalized
                inputs. This cannot be passed in if the module was constructed with
                ``create_scale=True``.
            offset: An array up to n-D. The shape of this tensor must be broadcastable
                to the shape of ``inputs``. This is the offset applied to the normalized
                inputs. This cannot be passed in if the module was constructed with
                ``create_offset=True``.

        Returns:
            The array, normalized across all but the last dimension.
        """
        inputs = jnp.asarray(inputs, jnp.float32)

        if training is None:
            training = self.is_training()

        if self.create_scale and scale is not None:
            raise ValueError("Cannot pass `scale` at call time if `create_scale=True`.")
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

        if training or test_local_stats or not hasattr(self.mean_ema, "average"):
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

        if training or not hasattr(self.mean_ema, "average"):
            self.mean_ema(mean)
            self.var_ema(var)

        w_shape = [1 if i in axis else inputs.shape[i] for i in range(inputs.ndim)]
        w_dtype = jnp.float32

        if self.create_scale:
            scale = self.add_parameter(
                "scale", lambda: self.scale_init(w_shape, w_dtype)
            )
        elif scale is None:
            scale = np.ones([], dtype=w_dtype)

        if self.create_offset:
            offset = self.add_parameter(
                "offset", lambda: self.offset_init(w_shape, w_dtype)
            )
        elif offset is None:
            offset = np.zeros([], dtype=w_dtype)

        inv = scale * jax.lax.rsqrt(var + self.eps)
        output = (inputs - mean) * inv + offset
        return jnp.asarray(output, self.dtype)
