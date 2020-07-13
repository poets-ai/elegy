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


    ```python
    model = elegy.Model(
        model_fn,
        optimizer=optix.rmsprop(0.01)
        loss=lambda: [elegy.losses.CategoricalCrossentropy()],
        metrics=lambda: [elegy.metrics.Accuracy()],
    )
    ```
    model.compile(optimizer=elegy.optimizers.RMSprop(0.01),
                    loss=elegy.losses.CategoricalCrossentropy(),
                    metrics=[elegy.metrics.CategoricalAccuracy()])
    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.batch(32)
    model.fit(dataset, epochs=10)
    ```
    To be implemented by subclasses:
    * `__init__()`: All state variables should be created in this method by
        calling `self.add_weight()` like: `self.var = self.add_weight(...)`
    * `update_state()`: Has all updates to the state variables like:
        self.var.assign_add(...).
    * `result()`: Computes and returns a value for the metric
        from the state variables.
    Example subclass implementation:
    ```python
    class BinaryTruePositives(elegy.metrics.Metric):
        def __init__(self, name='binary_true_positives', **kwargs):
        super(BinaryTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)
        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_weights(sample_weight, values)
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))
        def result(self):
        return self.true_positives
    ```
    """

    def __init__(
        self, name: tp.Optional[str] = None, dtype: tp.Optional[jnp.dtype] = None
    ):
        super().__init__(name=name)

        self._dtype = self._dtype = dtype if dtype is not None else jnp.float32
        self.call = utils.inject_dependencies(self.call)

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    @abstractmethod
    def call(self, *args, **kwargs):
        ...
