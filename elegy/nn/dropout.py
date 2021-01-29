# Implementation based on Tensorflow Keras and Haiku
# Tensorflow: https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/layers/core.py#L127-L224
# Haiku: https://github.com/deepmind/dm-haiku/blob/master/haiku/_src/basic.py#L293#L315

import typing as tp

import haiku as hk
import jax.numpy as jnp
import numpy as np

from elegy import module
from elegy import hooks
from elegy.module import Module


class Dropout(Module):
    """
    Applies Dropout to the input.

    The Dropout layer randomly sets input units to 0 with a frequency of `rate`
    at each step during training time, which helps prevent overfitting.
    Inputs not set to 0 are scaled up by `1/(1 - rate)` such that the sum over
    all inputs is unchanged.

    Note that the Dropout layer only applies when `training` is set to `True`
    such that no values are dropped during inference. When using `model.fit`,
    `training` will be appropriately set to True automatically, and in other
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

    outputs = dropout(data, training=True)

    print(outputs)
    # [[ 0.    1.25]
    # [ 2.5   3.75]
    # [ 5.    6.25]
    # [ 7.5   0.  ]
    # [10.    0.  ]]
    ```
    """

    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def call(
        self,
        x: jnp.ndarray,
        training: tp.Optional[bool] = None,
        rng: tp.Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Arguments:
            x: The value to be dropped out.
            training: Whether training is currently happening.
            rng: Optional RNGKey.
        Returns:
            x but dropped out and scaled by `1 / (1 - rate)`.
        """
        if training is None:
            training = self.is_training()

        return hk.dropout(
            rng=rng if rng is not None else self.next_key(),
            rate=self.rate if training else 0.0,
            x=x,
        )
