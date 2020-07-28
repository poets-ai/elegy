from elegy.module import Deferable
from elegy import types
import typing as tp
import haiku as hk
import jax.numpy as jnp
from abc import abstractmethod
from elegy import utils


class Metric(hk.Module, Deferable):
    """
    Encapsulates metric logic and state.

    Usage:

    ```python
    m = SomeMetric(...)
    for input in ...:
        result = m(input)
    print('Final result: ', result)
    ```

    Usage with the Model API:

    ```python
    class MLP(elegy.Module):
        def __apply__(self, image: jnp.ndarray) -> jnp.ndarray:
            mlp = hk.Sequential([
                hk.Flatten(),
                hk.Linear(300),
                jax.nn.relu,
                hk.Linear(10),
            ])
            return mlp(image)

    model = elegy.Model(
        module=MLP.defer(),
        loss=[
            elegy.losses.SparseCategoricalCrossentropy(from_logits=True),
        ],
        metrics=[
            elegy.metrics.SparseCategoricalAccuracy.defer()
        ],
        optimizer=optix.rmsprop(1e-3),
    )
    ```

    To be implemented by subclasses:

    * `__apply__()`: All state variables should be created in this method by
        calling `haiku.get_state()`, update this state by calling
        `haiku.set_state(...)`, and return a result based on these states.

    Example subclass implementation:

    ```python
    class Accuracy(elegy.Metric):
        def __apply__(self, y_true, y_pred):

            total = hk.get_state("total", [], jnp.zeros)
            count = hk.get_state("count", [], jnp.zeros)

            total += jnp.sum(y_true == y_pred)
            count += jnp.prod(y_true.shape)

            hk.set_state("total", total)
            hk.set_state("count", count)

            return total / count
    ```
    """

    def __init__(
        self,
        name: tp.Optional[str] = None,
        dtype: tp.Optional[jnp.dtype] = None,
        on: tp.Optional[types.IndexLike] = None,
    ):
        """
        Base Metric constructor.

        Arguments:
            name: string name of the metric instance.
            dtype: data type of the metric result.
            on: A string or integer, or iterable of string or integers, that
                indicate how to index/filter the `y_true` and `y_pred`
                arguments before passing them to `__apply__`. For example if `on = "a"` then
                `y_true = y_true["a"]`. If `on` is an iterable
                the structures will be indexed iteratively, for example if `on = ["a", 0, "b"]`
                then `y_true = y_true["a"][0]["b"]`, same for `y_pred`. For more information
                check out [Keras-like behavior](https://poets-ai.github.io/elegy/guides/modules-losses-metrics/#keras-like-behavior).
        """

        super().__init__(name=name)

        self._dtype = self._dtype = dtype if dtype is not None else jnp.float32
        self._labels_filter = (on,) if isinstance(on, (str, int)) else on
        self.__apply__ = utils.inject_dependencies(self.__apply__)

    def __call__(self, y_true=None, y_pred=None, **kwargs):

        if self._labels_filter is not None:
            if y_true is not None:
                for index in self._labels_filter:
                    y_true = y_true[index]

            if y_pred is not None:
                for index in self._labels_filter:
                    y_pred = y_pred[index]

        return self.__apply__(y_true=y_true, y_pred=y_pred, **kwargs)

    @abstractmethod
    def __apply__(self, *args, **kwargs):
        ...
