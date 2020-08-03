import typing as tp

import haiku as hk
import numpy as np

from elegy import utils
from elegy.module import Module


class Sequential(Module):
    """
    Sequentially calls the given list of layers.

    Note that ``Sequential`` is limited in the range of possible architectures
    it can handle. This is a deliberate design decision; ``Sequential`` is only
    meant to be used for the simple case of fusing together modules/ops where
    the input of a particular module/op is the output of the previous one.

    Another restriction is that it is not possible to have extra arguments in the
    ``__call__`` method that are passed to the constituents of the module - for
    example, if there is a ``BatchNorm`` module in ``Sequential`` and the user
    wishes to switch the ``is_training`` flag. If this is the desired use case,
    the recommended solution is to subclass :class:`Module` and implement
    ``__call__``:

        >>> class CustomModule(hk.Module):
        ...   def __call__(self, x, is_training):
        ...     x = hk.Conv2D(32, 4, 2)(x)
        ...     x = hk.BatchNorm(True, True, 0.9)(x, is_training)
        ...     x = jax.nn.relu(x)
        ...     return x
    """

    def __init__(
        self,
        layers: tp.Iterable[tp.Callable[..., tp.Any]],
        name: tp.Optional[str] = None,
    ):
        super().__init__(name=name)
        self.layers = tuple(layers)

    def call(self, inputs, *args, **kwargs):
        """Connects all layers. *args and **kwargs are passed to the first layer."""
        out = inputs
        for i, layer in enumerate(self.layers):
            if i == 0:
                out = layer(out, *args, **kwargs)
            else:
                out = layer(out)
        return out

