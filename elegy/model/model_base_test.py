import elegy
import jax.numpy as jnp
import numpy as np
import unittest


class TestModelBase(unittest.TestCase):
    def test_predict(self):
        class Model(elegy.model.model_base.ModelBase):
            def pred_step(self, x, net_states, initializing):
                if initializing:
                    states = elegy.States(net_states=0)
                else:
                    states = elegy.States(net_states=net_states + 1)

                return elegy.PredStep.simple(x + 1.0, states)

        model = Model()

        x = np.random.uniform(size=(100, 1))
        y = model.predict(x, batch_size=50)

        assert np.allclose(y, x + 1.0)
        assert model.states.net_states == 2
        assert model.initial_states.net_states == 0

    def test_evaluate(self):
        class Model(elegy.model.model_base.ModelBase):
            def test_step(self, x, metrics_states, initializing):
                if initializing:
                    states = elegy.States(metrics_states=0)
                else:
                    states = elegy.States(metrics_states=metrics_states + 1)

                return elegy.TestStep(
                    loss=0.1,
                    logs=dict(loss=jnp.sum(x)),
                    states=states,
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

    def test_fit(self):
        class Model(elegy.model.model_base.ModelBase):
            def train_step(self, x, optimizer_states, initializing):
                if initializing:
                    states = elegy.States(optimizer_states=0)
                else:
                    states = elegy.States(optimizer_states=optimizer_states + 1)

                return elegy.TrainStep(
                    logs=dict(loss=jnp.sum(x)),
                    states=states,
                )

        model = Model()

        x = np.random.uniform(size=(100, 1))

        history = model.fit(x, batch_size=100)
        assert np.allclose(history.history["loss"], np.sum(x))
        assert model.states.optimizer_states == 1

        history = model.fit(x, batch_size=50, shuffle=False)
        assert np.allclose(history.history["loss"], np.sum(x[50:]))
        assert model.states.optimizer_states == 3
