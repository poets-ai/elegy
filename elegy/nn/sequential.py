from elegy.module import Deferable
from elegy import hooks
import typing as tp

import haiku as hk
import numpy as np


class Sequential(hk.Sequential, Deferable):
    """
    Sequentially calls the given list of layers.

    Note that `Sequential` is limited in the range of possible architectures
    it can handle. This is a deliberate design decision; `Sequential` is only
    meant to be used for the simple case of fusing together modules/ops where
    the input of a particular module/op is the output of the previous one.

    Another restriction is that it is not possible to have extra arguments in the
    `__call__` method that are passed to the constituents of the module - for
    example, if there is a `BatchNorm` module in `Sequential` and the user
    wishes to switch the `is_training` flag. If this is the desired use case,
    the recommended solution is to subclass :class:`Module` and implement
    `__call__`:

    ```python
    class CustomModule(elegy.Module):
        def __call__(self, x, is_training):
            x = elegy.Conv2D(32, 4, 2)(x)
            x = elegy.BatchNorm(True, True, 0.9)(x, is_training)
            x = jax.nn.relu(x)
            return x
    ```
    """

    def __init__(
        self,
        layers: tp.Iterable[tp.Callable[..., tp.Any]],
        name: tp.Optional[str] = None,
    ):
        """
        Creates a Sequential instance.

        Arguments:
            layers: A list of moules or functions that take a single input.
            name: An optional string name for the class. Must be a valid Python
                identifier. If `name` is not provided then the class name for the
                current instance is converted to `lower_snake_case` and used instead.
        """
        super().__init__(layers=layers, name=name)

    def __call__(self, inputs: np.ndarray, *args, **kwargs):
        """
        Arguments:
            inputs: Input array.
            args: Additional positional input arguments for the first layer.
            kwargs: Additional keyword arguments for the first layer.
        """
        outputs = super().__call__(inputs, *args, **kwargs)

        hooks.add_summary(None, self.__class__.__name__, outputs)

        return outputs
