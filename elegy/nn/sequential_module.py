import functools
import typing as tp

import haiku as hk
import numpy as np

from elegy import utils, hooks
from elegy.module import Module, LOCAL, Context


def sequential(*layers: tp.Callable[..., tp.Any]) -> tp.Callable[..., tp.Any]:
    """
    Connects all layers. *args and **kwargs are passed to the first layer.

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

        if LOCAL.contexts:
            context: Context = LOCAL.contexts[-1]

            if not context.module_c:
                raise ValueError(
                    "Cannot execute `sequential` outside of a module's `call` or `init`."
                )

            module: Module = context.module_c[-1]

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
                    hooks.add_summary(name, out)
            return out

        else:
            raise ValueError(
                "Cannot execute `sequential` outside of an `elegy.context`"
            )

    return call


class Sequential(Module):
    """
    Sequentially calls the given list of layers.

    Note that `Sequential` is limited in the range of possible architectures
    it can handle. This is a deliberate design decision; `Sequential` is only
    meant to be used for the simple case of fusing together modules/ops where
    the input of a particular module/op is the output of the previous one.

    Another restriction is that it is not possible to have extra arguments in the
    `call` method that are passed to the constituents of the module - for
    example, if there is a `BatchNorm` module in `Sequential` and the user
    wishes to switch the `training` flag. If this is the desired use case,
    the recommended solution is to subclass `Module` and implement
    `call`:

    ```python
    class CustomModule(elegy.Module):
        def call(self, x, training):
            x = elegy.nn.Conv2D(32, 4, 2)(x)
            x = elegy.nn.BatchNorm(True, True, 0.9)(x, training)
            x = jax.nn.relu(x)
            return x
    ```
    """

    def __init__(
        self, layers: tp.Callable[[], tp.Iterable[tp.Callable[..., tp.Any]]], **kwargs
    ):
        super().__init__(**kwargs)
        self.layers = tuple(layers())

        # set signature of call to the signature of of the first layer
        # by creating a wrapper function.
        current_call = self.call

        @utils.wraps(self.layers[0])
        def call(*args, **kwargs):
            return current_call(*args, **kwargs)

        self.call = call

    def call(self, *args, **kwargs):
        """Connects all layers. *args and **kwargs are passed to the first layer."""
        return sequential(*self.layers)(*args, **kwargs)
