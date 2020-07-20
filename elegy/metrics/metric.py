from elegy import types
from elegy.utils import Deferable
import typing as tp
import haiku as hk
import jax.numpy as jnp
from abc import abstractmethod
from elegy import utils


class Metric(hk.Module, Deferable):
    """Encapsulates metric logic and state.

Usage:
```python
m = SomeMetric(...)
for input in ...:
    m.update_state(input)
print('Final result: ', m.result().numpy())
```
Usage with elegy API:
```python
import haiku as hk
import jax
from jax.experimental import optix

def module_fn(x):
    return hk.Sequential([
        hk.Linear(64), 
        jax.nn.relu,
        hk.Linear(64), 
        jax.nn.relu,
        hk.Linear(10), 
        jax.nn.softmax,
    ])(x)

model = elegy.Model(
    module_fn,
    optimizer=optix.rmsprop(0.01)
    loss=elegy.losses.CategoricalCrossentropy(),
    metrics=elegy.metrics.Accuracy.defer(),
)
```
To be implemented by subclasses:

- `call()`: Computes the actual metric

Example subclass implementation:

- TODO
    """

    def __init__(
        self,
        name: tp.Optional[str] = None,
        dtype: tp.Optional[jnp.dtype] = None,
        on: tp.Optional[types.IndexLike] = None,
    ):
        super().__init__(name=name)

        self._dtype = self._dtype = dtype if dtype is not None else jnp.float32
        self._labels_filter = (on,) if isinstance(on, (str, int)) else on
        self.call = utils.inject_dependencies(self.call)

    def __call__(self, y_true=None, y_pred=None, **kwargs):

        if self._labels_filter is not None:
            if y_true is not None:
                for index in self._labels_filter:
                    y_true = y_true[index]

            if y_pred is not None:
                for index in self._labels_filter:
                    y_pred = y_pred[index]

        return self.call(y_true=y_true, y_pred=y_pred, **kwargs)

    @abstractmethod
    def call(self, *args, **kwargs):
        ...
