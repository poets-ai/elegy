import unittest

import elegy as eg
import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from tests.model.model_core_test import ModelCoreTest


class TestModelBase(unittest.TestCase):
    def test_predict(self):

        N = 0

        class Model(eg.ModelBase):
            a: jnp.ndarray = eg.node()

            def init_step(
                self,
                key: jnp.ndarray,
            ) -> "Model":
                self.a = jnp.array(0, dtype=jnp.int32)
                return self

            def pred_step(self, inputs):
                nonlocal N
                N += 1

                preds = inputs + 1.0
                self.a += 1

                return preds, self

            def reset_metrics(self):
                pass

        model = Model()

        x = np.random.uniform(size=(100, 1))
        y = model.predict(x, batch_size=50)

        assert np.allclose(y, x + 1.0)
        assert model.a == 2
        assert N == 1

        y = model.predict(x, batch_size=50)
        assert np.allclose(y, x + 1.0)
        assert model.a == 4
        assert N == 1

        model.eager = True

        y = model.predict(x, batch_size=50)
        assert np.allclose(y, x + 1.0)
        assert model.a == 6
        assert N == 3

    def test_evaluate(self):
        N = 0

        class Model(eg.ModelBase):
            a: jnp.ndarray = eg.node()

            def init_step(
                self,
                key: jnp.ndarray,
            ) -> "Model":
                self.a = jnp.array(0, dtype=jnp.int32)
                return self

            def test_step(self, inputs, labels):
                nonlocal N
                N += 1

                preds = inputs + 1.0
                self.a += 1

                loss = 0.1
                logs = dict(loss=jnp.sum(inputs))

                return loss, logs, self

            def reset_metrics(self):
                pass

        model = Model()

        x = np.random.uniform(size=(100, 1))

        logs = model.evaluate(x, batch_size=100)
        assert np.allclose(logs["loss"], np.sum(x))
        assert N == 1
        assert model.a == 1

        logs = model.evaluate(x, batch_size=50)
        assert np.allclose(logs["loss"], np.sum(x[50:]))
        assert N == 2
        assert model.a == 3

        logs = model.evaluate(x, batch_size=50)
        assert np.allclose(logs["loss"], np.sum(x[50:]))
        assert N == 2
        assert model.a == 5

        model.eager = True

        logs = model.evaluate(x, batch_size=50)
        assert np.allclose(logs["loss"], np.sum(x[50:]))
        assert N == 4
        assert model.a == 7

    def test_fit(self):
        N = 0

        class Model(eg.ModelBase):
            a: jnp.ndarray = eg.node()

            def init_step(
                self,
                key: jnp.ndarray,
            ) -> "Model":
                self.a = jnp.array(0, dtype=jnp.int32)
                return self

            def train_step(self, inputs, labels):
                nonlocal N
                N += 1

                self.a += 1

                logs = dict(loss=jnp.sum(inputs))

                return logs, self

            def reset_metrics(self):
                pass

        model = Model()

        x = np.random.uniform(size=(100, 1))

        history = model.fit(x, batch_size=100)
        assert np.allclose(history.history["loss"], np.sum(x))
        assert N == 1
        assert model.a == 1

        history = model.fit(x, batch_size=50, shuffle=False)
        assert np.allclose(history.history["loss"][0], np.sum(x[50:]))
        assert N == 2
        assert model.a == 3

        history = model.fit(x, batch_size=50, shuffle=False)
        assert np.allclose(history.history["loss"], np.sum(x[50:]))
        assert N == 2
        assert model.a == 5

        model.eager = True

        history = model.fit(x, batch_size=50, shuffle=False)
        assert np.allclose(history.history["loss"], np.sum(x[50:]))
        assert N == 4
        assert model.a == 7

    def test_init(self):
        class Model(eg.ModelBase):
            def init_step(self, x, y_true, states: eg.States):
                return states.update(a=x.shape, b=y_true.shape)

        model = Model()

        x = np.random.uniform(size=(10, 1))
        y = np.random.uniform(size=(10, 3))

        model.init(x=x, y=y, batch_size=2)

        assert model.states.a == (2, 1)
        assert model.states.b == (2, 3)
        assert model.initialized

    def test_init_dataloader(self):
        class Model(eg.ModelBase):
            def init_step(self, x, y_true, states: eg.States):
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
        class Model(eg.ModelBase):
            def init_step(self, x, states: eg.States):
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


if __name__ == "__main__":
    TestModelBase().test_fit()