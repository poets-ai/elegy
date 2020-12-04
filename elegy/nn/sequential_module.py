import functools
import typing as tp

import haiku as hk
import numpy as np

from elegy import utils
from elegy.module import Module, LOCAL, LocalContext, add_summary, get_module
from elegy import module


def sequential(*layers: tp.Callable[..., tp.Any]) -> tp.Callable[..., tp.Any]:
    """
    Connects all layers. `*args` and `**kwargs` are passed to the first layer.

    ```python
    def call(self, x):
        mlp = elegy.nn.sequential(
            elegy.nn.Linear(64),
            jax.nn.relu,
            elegy.nn.Linear(32),
            jax.nn.relu,
            elegy.nn.Linear(10),
            jax.nn.softmax,
        )
        y = mlp(x)
        ...
    ```

    !!! Note
        `sequential` is not a `Module`, that is, it wont create a scope over the layers it runs,
        in constrast to `Sequential` layers are eagerly instantiate outside of `sequential`
        and just passed to it to automate the execution.

    Arguments:
        layers: Modules or functions passed as `*args`

    Returns:
        A callable that waits for the inputs and applies the layers sequentially.
    """

    def call(inputs, *args, **kwargs):

        out = inputs
        for i, layer in enumerate(layers):
            if i == 0:
                out = layer(out, *args, **kwargs)
            else:
                out = layer(out)

            if not isinstance(layer, Module):
                name = (
                    layer.__name__
                    if hasattr(layer, "__name__")
                    else layer.__class__.__name__
                )
                add_summary(name, out)
        return out

    return call


class Sequential(Module):
    """
    Creates a Module from a zero argument lambda that produces a list of Modules or function to be executed sequentially. The lambda is necessary so that all sub-modules are instantiated inside the context of the Sequential module.

    ```python
    mlp = elegy.nn.Sequential(
        lambda: [
            elegy.nn.Linear(64),
            jax.nn.relu,
            elegy.nn.Linear(32),
            jax.nn.relu,
            elegy.nn.Linear(10),
            jax.nn.softmax,
        ]
    )
    y = mlp(x)
    ```
    """

    def __init__(
        self, layers: tp.Callable[[], tp.Iterable[tp.Callable[..., tp.Any]]], **kwargs
    ):
        self.layers = tuple(layers())

        # set signature of call to the signature of of the first layer
        # by creating a wrapper function.
        current_call = self.call

        @utils.wraps(self.layers[0])
        def call(*args, **kwargs):
            return current_call(*args, **kwargs)

        self.call = call
        super().__init__(**kwargs)

    def call(self, *args, **kwargs):
        """Connects all layers. `*args` and `**kwargs` are passed to the first layer."""
        return sequential(*self.layers)(*args, **kwargs)
