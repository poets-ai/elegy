import elegy
import jax.numpy as jnp
import numpy as np


def test_predict():
    class Model(elegy.model.model_base.ModelBase):
        def init(self, mode: elegy.Mode):
            return elegy.States(net_states=0)

        def pred_step(self, x, net_states):
            return x + 1.0, elegy.States(net_states=net_states + 1)

        def test_step(self):
            ...

        def train_step(self):
            ...

    model = Model()

    x = np.random.uniform(size=(100, 1))
    y = model.predict(x, batch_size=50)

    assert np.allclose(y, x + 1.0)
    assert model.states.net_states == 2
    assert model.initial_states.net_states == 0


def test_evaluate():
    class Model(elegy.model.model_base.ModelBase):
        def init(self, mode: elegy.Mode):
            return elegy.States(metrics_states=0)

        def pred_step(self, x):
            return x + 1, elegy.States()

        def test_step(self, x, metrics_states):
            return elegy.Evaluation(
                loss=0.1,
                logs=dict(loss=jnp.sum(x)),
                states=elegy.States(metrics_states=metrics_states + 1),
            )

        def train_step(self):
            ...

    model = Model(run_eagerly=True)

    x = np.random.uniform(size=(100, 1))

    logs = model.evaluate(x, batch_size=100)
    assert np.allclose(logs["loss"], np.sum(x))
    assert model.states.metrics_states == 1

    logs = model.evaluate(x, batch_size=50)
    assert np.allclose(logs["loss"], np.sum(x[50:]))
    assert model.states.metrics_states == 2


def test_fit():
    class Model(elegy.model.model_base.ModelBase):
        def init(self, mode: elegy.Mode):
            return elegy.States(optimizer_states=0)

        def pred_step(self, x):
            ...

        def test_step(self, x, optimizer_states):
            ...

        def train_step(self, x, optimizer_states):
            return elegy.Training(
                logs=dict(loss=jnp.sum(x)),
                states=elegy.States(optimizer_states=optimizer_states + 1),
            )

    model = Model()

    x = np.random.uniform(size=(100, 1))

    history = model.fit(x, batch_size=100)
    assert np.allclose(history.history["loss"], np.sum(x))
    assert model.states.optimizer_states == 1

    history = model.fit(x, batch_size=50, shuffle=False)
    assert np.allclose(history.history["loss"], np.sum(x[50:]))
    assert model.states.optimizer_states == 3
