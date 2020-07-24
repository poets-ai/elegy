# Implementation based on Tensorflow Keras and Haiku
# Tensorflow: https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/layers/core.py#L127-L224
# Haiku: https://github.com/deepmind/dm-haiku/blob/master/haiku/_src/basic.py#L293#L315

import typing as tp

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from elegy.module import Module


class Dropout(Module):
    """
    Applies Dropout to the input.

    The Dropout layer randomly sets input units to 0 with a frequency of `rate`
    at each step during training time, which helps prevent overfitting.
    Inputs not set to 0 are scaled up by `1/(1 - rate)` such that the sum over
    all inputs is unchanged.

    Note that the Dropout layer only applies when `is_training` is set to `True`
    such that no values are dropped during inference. When using `model.fit`,
    `is_training` will be appropriately set to True automatically, and in other
    contexts, you can set the kwarg explicitly to True when calling the layer.

    ### Example
    ```python
    dropout = elegy.nn.Dropout(0.2)
    data = np.arange(10).reshape(5, 2).astype(np.float32)
    
    print(data)
    # [[0. 1.]
    # [2. 3.]
    # [4. 5.]
    # [6. 7.]
    # [8. 9.]]

    outputs = dropout(data, is_training=True)
    
    print(outputs)
    # [[ 0.    1.25]
    # [ 2.5   3.75]
    # [ 5.    6.25]
    # [ 7.5   0.  ]
    # [10.    0.  ]]
    ```
    """

    def __init__(self, rate, name: tp.Optional[str] = None):
        super().__init__(name=name)
        self.rate = rate

    def call(
        self, x: np.ndarray, is_training: bool, rng: tp.Optional[np.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Arguments:
            x: The value to be dropped out.
            is_training: Whether training is currently happening.
            rng: Optional RNGKey.
        Returns:
            x but dropped out and scaled by `1 / (1 - rate)`.
        """
        return hk.dropout(
            rng=hk.next_rng_key() if rng is None else rng,
            rate=self.rate if is_training else 0.0,
            x=x,
        )
