import typing as tp
from abc import abstractmethod

import jax.numpy as jnp
from elegy import module, types, utils


class Metric(module.Module):
    """
    Encapsulates metric logic and state. Metrics accumulate state between `apply`s such
    that their output value reflect the metric as if calculated on the whole data
    given up to that point.

    Usage:

    ```python
    m = SomeMetric()
    _, state = m.init()(x)
    for x in batch:
        result = m.apply(state)(x)
    print('Final result: ', result)
    ```

    Usage with the Model API:

    ```python
    >>> import elegy, jax, optax
    >>> model = elegy.Model(
    ...     module=elegy.nn.Sequential(
    ...         lambda: [
    ...             elegy.nn.Flatten(),
    ...             elegy.nn.Linear(300),
    ...             jax.nn.relu,
    ...             elegy.nn.Linear(10),
    ...         ]
    ...     ),
    ...     loss=[
    ...         elegy.losses.SparseCategoricalCrossentropy(from_logits=True),
    ...     ],
    ...     metrics=[
    ...         elegy.metrics.SparseCategoricalAccuracy()
    ...     ],
    ...     optimizer=optax.rmsprop(1e-3),
    ... )

    ```

    To be implemented by subclasses:

    * `call()`: All state variables should be created in this method by
        calling `self.add_parameter(..., trainable=False)`, update this state by calling
        `self.update_parameter(...)`, and return a result based on these states.

    Example subclass implementation:

    ```python
    >>> class Accuracy(elegy.Metric):
    ...    def call(self, y_true, y_pred):
    ...
    ...        total = self.add_parameter("total", lambda: jnp.array(0), trainable=False)
    ...        count = self.add_parameter("count", lambda: jnp.array(0), trainable=False)
    ...
    ...        total += jnp.sum(y_true == y_pred)
    ...        count += jnp.prod(y_true.shape)
    ...
    ...        self.update_parameter("total", total)
    ...        self.update_parameter("count", count)
    ...
    ...        return total / count

    ```
    """

    __all__ = ["__init__", "call"]

    def __init__(self, on: tp.Optional[types.IndexLike] = None, **kwargs):
        """
        Base Metric constructor.

        Arguments:
            on: A string or integer, or iterable of string or integers, that
                indicate how to index/filter the `y_true` and `y_pred`
                arguments before passing them to `call`. For example if `on = "a"` then
                `y_true = y_true["a"]`. If `on` is an iterable
                the structures will be indexed iteratively, for example if `on = ["a", 0, "b"]`
                then `y_true = y_true["a"][0]["b"]`, same for `y_pred`. For more information
                check out [Keras-like behavior](https://poets-ai.github.io/elegy/guides/modules-losses-metrics/#keras-like-behavior).
        """

        super().__init__(**kwargs)

        self._labels_filter = (on,) if isinstance(on, (str, int)) else on

    def __call__(self, *args, **kwargs):

        if self._labels_filter is not None:
            if "y_true" in kwargs and kwargs["y_true"] is not None:
                for index in self._labels_filter:
                    kwargs["y_true"] = kwargs["y_true"][index]

            if "y_pred" in kwargs and kwargs["y_pred"] is not None:
                for index in self._labels_filter:
                    kwargs["y_pred"] = kwargs["y_pred"][index]

        return super().__call__(*args, **kwargs)

    @abstractmethod
    def call(self, *args, **kwargs):
        ...
