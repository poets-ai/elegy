from elegy import utils
from elegy.module import Module
import typing as tp

import haiku as hk
import numpy as np


class Sequential(Module):
    """
    Sequentially calls the given list of layers.

    ```python
    class MLP(elegy.Module):
        def __apply__(self, x):
            mlp = elegy.nn.Sequential([
                elegy.nn.Flatten(),
                elegy.nn.Linear(100),
                jax.nn.relu,
                elegy.nn.Linear(10),
            ])
            return mlp(x)
    ```

    For convenience Elegy's Sequential also accept a lambda as argument which
    lets you easily pass simple modules to the Model API.

    ```python
    model = elegy.Model(
        module=elegy.nn.Sequential.defer(
            lambda:[
                elegy.nn.Flatten(),
                elegy.nn.Linear(100),
                jax.nn.relu,
                elegy.nn.Linear(10),
            ]
        ),
        loss=elegy.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=optix.adam(1e-3),
    )
    ```

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
        layers: tp.Union[
            tp.Iterable[tp.Callable[..., tp.Any]],
            tp.Callable[[], tp.Iterable[tp.Callable[..., tp.Any]]],
        ],
        name: tp.Optional[str] = None,
    ):
        """
        Creates a Sequential instance.

        Arguments:
            layers: A list of moules or functions that take a single input. You can
                also pass a cero parameter function of the previous.
            name: An optional string name for the class. Must be a valid Python
                identifier. If `name` is not provided then the class name for the
                current instance is converted to `lower_snake_case` and used instead.
        """
        super().__init__(name=name)
        self.layers = layers

    def __apply__(self, inputs: np.ndarray, *args, **kwargs):
        """
        Connects all layers. *args and **kwargs are passed to the first layer.
        
        Arguments:
            inputs: Input array.
            args: Additional positional input arguments for the first layer.
            kwargs: Additional keyword arguments for the first layer.
        """

        outputs = inputs

        for i, layer in enumerate(
            self.layers if isinstance(self.layers, tp.Iterable) else self.layers()
        ):
            if i == 0:
                outputs = utils.inject_dependencies(layer)(outputs, *args, **kwargs)
            else:
                outputs = layer(outputs)

        return outputs
