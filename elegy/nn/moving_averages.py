# Lint as: python3
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
"""Moving averages."""

import re

import jax
import jax.numpy as jnp
from haiku._src import data_structures
import numpy as np

from elegy import initializers, module, types


class ExponentialMovingAverage(module.Module):
    """Maintains an exponential moving average.

    This uses the Adam debiasing procedure.
    See https://arxiv.org/pdf/1412.6980.pdf for details.
    """

    counter: np.ndarray
    hidden: np.ndarray
    average: np.ndarray

    def __init__(self, decay, zero_debias=True, warmup_length=0, **kwargs):
        """Initializes an ExponentialMovingAverage module.

        Args:
            decay: The chosen decay. Must in [0, 1). Values close to 1 result in slow
                decay; values close to 0 result in fast decay.
            zero_debias: Whether to run with zero-debiasing.
            warmup_length: A positive integer, EMA has no effect until
                the internal counter has reached `warmup_length` at which point the
                initial value for the decaying average is initialized to the input value
                after `warmup_length` iterations.
            kwargs: Additional keyword arguments passed to Module.
        """
        super().__init__(**kwargs)
        self._decay = decay
        if warmup_length < 0:
            raise ValueError(
                f"`warmup_length` is {warmup_length}, but should be non-negative."
            )
        self._warmup_length = warmup_length
        self._zero_debias = zero_debias
        if warmup_length and zero_debias:
            raise ValueError(
                "Zero debiasing does not make sense when warming up the value of the "
                "average to an initial value. Set zero_debias=False if setting "
                "warmup_length to a non-zero value."
            )

    def _cond(self, cond, t, f, dtype):
        """Internal, implements jax.lax.cond without control flow."""
        c = cond.astype(dtype)
        return c * t + (1.0 - c) * f

    def initialize(self, value):
        """If uninitialized sets the average to ``zeros_like`` the given value."""
        self.add_parameter(
            "hidden", lambda: jnp.zeros(value.shape, jnp.float32), trainable=False
        )
        self.add_parameter(
            "average", lambda: jnp.zeros(value.shape, jnp.float32), trainable=False
        )

    def call(self, value: jnp.ndarray, update_stats=True):
        """Updates the EMA and returns the new value.

        Args:
          value: The array-like object for which you would like to perform an
            exponential decay on.
          update_stats: A Boolean, whether to update the internal state
            of this object to reflect the input value. When `update_stats` is False
            the internal stats will remain unchanged.

        Returns:
          The exponentially weighted average of the input value.
        """
        if not isinstance(value, jnp.ndarray):
            value = jnp.asarray(value)

        counter = self.add_parameter(
            "counter",
            lambda: jnp.broadcast_to(-self._warmup_length, ()).astype(jnp.int32),
            trainable=False,
        )
        counter += 1

        decay = jax.lax.convert_element_type(self._decay, value.dtype)
        if self._warmup_length > 0:
            decay = self._cond(counter <= 0, 0.0, decay, value.dtype)

        one = jnp.ones([], value.dtype)
        hidden = self.add_parameter(
            "hidden",
            lambda: jnp.zeros(value.shape, jnp.float32),
            trainable=False,
        )
        hidden = hidden * decay + value * (one - decay)

        average = hidden
        if self._zero_debias:
            average /= one - jnp.power(decay, counter)

        self.add_or_update_parameter("average", average, trainable=False)

        if update_stats:
            self.update_parameter("counter", counter)
            self.update_parameter("hidden", hidden)
            self.update_parameter("average", average)

        return average


class EMAParamsTree(module.Module):
    """Maintains an exponential moving average for all parameters in a tree.

    While ExponentialMovingAverage is meant to be applied to single parameters
    within a function, this class is meant to be applied to the entire tree of
    parameters for a function.

    Given a set of parameters for some network:

    ```python
    >>> import elegy
    >>> x = jnp.ones([1, 1])
    >>> linear = elegy.nn.Linear(10)
    >>> y, params = linear.init(rng=elegy.RNGSeq(42))(x)

    ```

    You might use the EMAParamsTree like follows:

    ```python
    >>> ema = elegy.nn.EMAParamsTree(0.2)
    >>> _, ema_state = ema.init()(params)
    >>> ema_params, ema_state = ema.apply(ema_state)(params)

    ```

    Here, we are transforming a Haiku function and constructing its parameters via
    an init_fn as normal, but are creating a second transformed function which
    expects a tree of parameters as input. This function is then called with
    the current parameters as input, which then returns an identical tree with
    every parameter replaced with its exponentially decayed average. This
    ema_params object can then be passed into the `network_fn` as usual, and will
    cause it to run with EMA weights.
    """

    def __init__(
        self, decay, zero_debias=True, warmup_length=0, ignore_regex="", **kwargs
    ):
        """Initializes an EMAParamsTree module.

        Args:
            decay: The chosen decay. Must in [0, 1). Values close to 1 result in slow
                decay; values close to 0 result in fast decay.
            zero_debias: Whether to run with zero-debiasing.
            warmup_length: A positive integer, EMA has no effect until
                the internal counter has reached `warmup_length` at which point the
                initial value for the decaying average is initialized to the input value
                after `warmup_length` iterations.
            ignore_regex: A string. Any parameter in the tree whose name matches this
                regex will not have any moving average applied to it. The empty string
                means this module will EMA all parameters.
            kwargs: Additional keyword arguments passed to Module.
        """
        super().__init__(**kwargs)
        self._decay = decay
        self._zero_debias = zero_debias
        self._warmup_length = warmup_length
        self._ignore_regex = ignore_regex

    def call(self, tree, update_stats=True):
        def maybe_ema(k, v):
            if self._ignore_regex and re.match(self._ignore_regex, k):
                return v
            else:
                ema_name = k.replace("/", "__").replace("~", "_tilde_")
                return ExponentialMovingAverage(
                    self._decay, self._zero_debias, self._warmup_length, name=ema_name
                )(v, update_stats=update_stats)

        # We want to potentially replace params with EMA'd versions.
        new_values = {}
        for module_name, param_dict in tree.items():
            new_values[module_name] = {
                k: maybe_ema("/".join([module_name, k]), v)
                for k, v in param_dict.items()
            }
        return data_structures.to_immutable_dict(new_values)
