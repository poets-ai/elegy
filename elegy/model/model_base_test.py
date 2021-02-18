import unittest

import elegy
import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class TestModelBase(unittest.TestCase):
    def test_predict(self):
        class Model(elegy.model.model_base.ModelBase):
            def init_step(self, x, states):
                _, states = self.pred_step(x, True, states)
                return states

            def pred_step(self, x, initializing, states):
                if initializing:
                    states = elegy.States(net_states=0)
                else:
                    states = elegy.States(net_states=states.net_states + 1)

                return elegy.PredStep(x + 1.0, states)

        model = Model()

        x = np.random.uniform(size=(100, 1))
        model.init(x, batch_size=50)
        y = model.predict(x, batch_size=50)

        assert np.allclose(y, x + 1.0)
        assert model.states.net_states == 2
        assert model.initial_states.net_states == 0

    def test_evaluate(self):
        class Model(elegy.model.model_base.ModelBase):
            def init_step(self, x, states):
                _, _, states = self.test_step(x, True, states)
                return states

            def test_step(self, x, initializing, states):
                if initializing:
                    states = elegy.States(metrics_states=0)
                else:
                    states = elegy.States(metrics_states=states.metrics_states + 1)

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
            def init_step(self, x, states):
                _, states = self.train_step(x, states, True)
                return states

            def train_step(self, x, states, initializing):
                if initializing:
                    states = elegy.States(optimizer_states=0)
                else:
                    states = elegy.States(optimizer_states=states.optimizer_states + 1)

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

    def test_init(self):
        class Model(elegy.model.model_base.ModelBase):
            def init_step(self, x, y_true, states: elegy.States):
                return states.update(a=x.shape, b=y_true.shape)

        model = Model()

        x = np.random.uniform(size=(10, 1))
        y = np.random.uniform(size=(10, 3))

        model.init(x=x, y=y, batch_size=2)

        assert model.states.a == (2, 1)
        assert model.states.b == (2, 3)
        assert model.initialized

    def test_init_dataloader(self):
        class Model(elegy.model.model_base.ModelBase):
            def init_step(self, x, y_true, states: elegy.States):
                return states.update(a=x.shape, b=y_true.shape)

            def pred_step(self, x, states):
                states = states.update(c=3)
                return x + 1, states

        model = Model()

        x = np.random.uniform(size=(10, 1))
        y = np.random.uniform(size=(10, 3))

        dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        dataloader = DataLoader(dataset, batch_size=2)

        model.init(x=dataloader, batch_size=2)

        assert model.states.a == (2, 1)
        assert model.states.b == (2, 3)
        assert model.initialized

        y_pred = model.predict(x=dataloader)
        assert jnp.allclose(y_pred, x + 1)
        y_pred = model.predict(x=dataloader)
        assert jnp.allclose(y_pred, x + 1)
        y_pred

    def test_init_predict(self):
        class Model(elegy.model.model_base.ModelBase):
            def init_step(self, x, states: elegy.States):
                return states.update(a=x.shape)

            def pred_step(self, x, states):
                states = states.update(c=3)
                return x + 1, states

        model = Model()

        x = np.random.uniform(size=(10, 1))

        model.init(x=x, batch_size=2)
        y_pred = model.predict(x=x, batch_size=2)

        assert jnp.allclose(y_pred, x + 1)
        assert model.states.a == (2, 1)
        assert model.states.c == 3
        assert model.initialized
