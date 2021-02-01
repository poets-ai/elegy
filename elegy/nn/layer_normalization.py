# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Layer Norm."""

import collections
from elegy import types
from typing import Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
from haiku._src import utils as hk_utils

# If you are forking replace this with `import haiku as hk`.
from elegy import initializers, module


class LayerNormalization(module.Module):
    """LayerNorm module.

    See: https://arxiv.org/abs/1607.06450.
    """

    def __init__(
        self,
        axis: Union[int, Sequence[int], slice] = -1,
        create_scale: bool = True,
        create_offset: bool = True,
        eps: float = 1e-5,
        scale_init: Optional[types.Initializer] = None,
        offset_init: Optional[types.Initializer] = None,
        **kwargs
    ):
        """Constructs a LayerNorm module.

        Args:
            axis: Integer, list of integers, or slice indicating which axes to
                normalize over.
            create_scale: Bool, defines whether to create a trainable scale
                per channel applied after the normalization.
            create_offset: Bool, defines whether to create a trainable offset
                per channel applied after normalization and scaling.
            eps: Small epsilon to avoid division by zero variance. Defaults ``1e-5``,
                as in the paper and Sonnet.
            scale_init: Optional initializer for gain (aka scale). By default, one.
            offset_init: Optional initializer for bias (aka offset). By default, zero.
            kwargs: Additional keyword arguments passed to Module.
        """
        super().__init__(**kwargs)
        if not create_scale and scale_init is not None:
            raise ValueError("Cannot set `scale_init` if `create_scale=False`.")
        if not create_offset and offset_init is not None:
            raise ValueError("Cannot set `offset_init` if `create_offset=False`.")

        if isinstance(axis, slice):
            self.axis = axis
        elif isinstance(axis, int):
            self.axis = (axis,)
        elif isinstance(axis, collections.Iterable) and all(
            isinstance(ax, int) for ax in axis
        ):
            self.axis = tuple(axis)
        else:
            raise ValueError("`axis` should be an int, slice or iterable of ints.")

        self.eps = eps
        self.create_scale = create_scale
        self.create_offset = create_offset
        self.scale_init = scale_init or jnp.ones
        self.offset_init = offset_init or jnp.zeros

    def call(
        self,
        inputs: jnp.ndarray,
        scale: Optional[jnp.ndarray] = None,
        offset: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Connects the layer norm.

        Args:
          inputs: An array, where the data format is ``[N, ..., C]``.
          scale: An array up to n-D. The shape of this tensor must be broadcastable
            to the shape of ``inputs``. This is the scale applied to the normalized
            inputs. This cannot be passed in if the module was constructed with
            ``create_scale=True``.
          offset: An array up to n-D. The shape of this tensor must be broadcastable
            to the shape of ``inputs``. This is the offset applied to the normalized
            inputs. This cannot be passed in if the module was constructed with
            ``create_offset=True``.

        Returns:
          The array, normalized.
        """
        if self.create_scale and scale is not None:
            raise ValueError("Cannot pass `scale` at call time if `create_scale=True`.")
        if self.create_offset and offset is not None:
            raise ValueError(
                "Cannot pass `offset` at call time if `create_offset=True`."
            )

        axis = self.axis
        if isinstance(axis, slice):
            axis = tuple(range(inputs.ndim)[axis])

        mean = jnp.mean(inputs, axis=axis, keepdims=True)
        variance = jnp.var(inputs, axis=axis, keepdims=True)

        param_shape = inputs.shape[-1:]
        if self.create_scale:
            scale = self.add_parameter(
                "scale", param_shape, jnp.float32, initializer=self.scale_init
            )
        elif scale is None:
            scale = np.array(1.0, dtype=inputs.dtype)

        if self.create_offset:
            offset = self.add_parameter(
                "offset", param_shape, jnp.float32, initializer=self.offset_init
            )
        elif offset is None:
            offset = np.array(0.0, dtype=inputs.dtype)

        scale = jnp.broadcast_to(scale, inputs.shape)
        offset = jnp.broadcast_to(offset, inputs.shape)
        mean = jnp.broadcast_to(mean, inputs.shape)

        inv = scale * jax.lax.rsqrt(variance + self.eps)
        return inv * (inputs - mean) + offset


class InstanceNormalization(LayerNormalization):
    """Normalizes inputs along the spatial dimensions.

    See `LayerNorm` for more details.
    """

    def __init__(
        self,
        create_scale: bool = True,
        create_offset: bool = True,
        eps: float = 1e-5,
        scale_init: Optional[types.Initializer] = None,
        offset_init: Optional[types.Initializer] = None,
        data_format: str = "channels_last",
        **kwargs
    ):
        """Constructs an `InstanceNormalization` module.

        This method creates a module which normalizes over the spatial dimensions.

        Args:
            create_scale: ``bool`` representing whether to create a trainable scale
                per channel applied after the normalization.
            create_offset: ``bool`` representing whether to create a trainable offset
                per channel applied after normalization and scaling.
            eps: Small epsilon to avoid division by zero variance. Defaults to
                ``1e-5``.
            scale_init: Optional initializer for the scale variable. Can only be set
                if ``create_scale=True``. By default scale is initialized to ``1``.
            offset_init: Optional initializer for the offset variable. Can only be set
                if ``create_offset=True``. By default offset is initialized to ``0``.
            data_format: The data format of the input. Can be either
                ``channels_first``, ``channels_last``, ``N...C`` or ``NC...``. By
                default it is ``channels_last``.
            kwargs: Additional keyword arguments passed to Module.
        """
        if hk_utils.get_channel_index(data_format) == 1:
            axis = slice(2, None)
        else:  # channel_index = -1
            axis = slice(1, -1)
        super().__init__(
            axis=axis,
            create_scale=create_scale,
            create_offset=create_offset,
            eps=eps,
            scale_init=scale_init,
            offset_init=offset_init,
            **kwargs
        )
