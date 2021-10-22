import typing as tp
import unittest

import elegy
import jax
import jax.numpy as jnp
import numpy as np
import treex as tx
from elegy.model.model_core import ModelCore


class ModelCoreTest(unittest.TestCase):
    def test_init(self):
        N = 0

        class Model(ModelCore):
            net_params: tx.Parameter[tp.Any]
            net_states: tx.State[tp.Any]
            metrics_states: tx.Metric[tp.Any]
            optimizer_states: tx.OptState[tp.Any]

            def __init__(self):
                super().__init__()
                self.net_params = None
                self.net_states = None
                self.metrics_states = None
                self.optimizer_states = None

            def pred_step(self) -> elegy.PredStep["Model"]:
                nonlocal N
                N = N + 1

                self.net_params = 1
                self.net_states = 2

                return None, self

            def test_step(self) -> elegy.TestStep["Model"]:
                _, model = self.pred_step()

                model.metrics_states = 3

                return 0, {}, model

            def train_step(self) -> elegy.TrainStep["Model"]:
                _, logs, model = self.test_step()

                model.optimizer_states = 3

                return logs, model

        model = Model()

        assert N == 0
        assert model.net_params is None
        assert model.net_states is None
        assert model.metrics_states is None
        assert model.optimizer_states is None

        model.init_on_batch(jax.random.PRNGKey(42))
        assert N == 1

        model.init_on_batch(jax.random.PRNGKey(42))
        assert N == 1

    def test_pred_step(self):
        class Model(ModelCore):
            def init_step(self, x, states):
                _, states = self.pred_step(x, states, True)
                return states

            def pred_step(self, x, states, initializing):
                if initializing:
                    states = elegy.States(net_states=0)
                else:
                    states = elegy.States(net_states=states.net_states + 1)

                return elegy.PredStep(
                    y_pred=1,
                    model=states,
                )

        model = Model()
        assert not hasattr(model.states, "net_states")

        model.init_on_batch(x=(np.array(1.0)))
        preds = model.predict_on_batch(inputs=(np.array(1.0)))
        assert preds == 1
        assert model.states.net_states == 1
        assert not hasattr(model.states, "net_params")

        model.eager = False

        preds = model.predict_on_batch(inputs=(np.array(1.0)))
        assert preds == 1
        assert model.states.net_states == 2

    def test_test_step(self):
        class Model(ModelCore):
            def init_step(self, states):
                _, _, states = self.test_step(states, True)
                return states

            def test_step(self, states, initializing):
                return elegy.TestStep(
                    loss=0.1,
                    logs=dict(loss=1.0),
                    model=elegy.States(metrics_states=0)
                    if initializing
                    else elegy.States(metrics_states=states.metrics_states + 1),
                )

            def train_step(self):
                ...

        model = Model()
        assert not hasattr(model.states, "metrics_states")

        logs = model.test_on_batch(inputs=(np.array(1.0)), labels=(1.0,))
        assert logs["loss"] == 1
        assert model.states.metrics_states == 1
        assert not hasattr(model.states, "net_params")

        model.eager = False

        logs = model.test_on_batch(inputs=(np.array(1.0)), labels=(1.0,))
        assert logs["loss"] == 1
        assert model.states.metrics_states == 2

    def test_train_step(self):
        class Model(ModelCore):
            def init_step(self, states):
                _, states = self.train_step(states, True)
                return states

            def train_step(self, states, initializing):
                return elegy.TrainStep(
                    logs=dict(loss=2.0),
                    model=elegy.States(optimizer_states=0)
                    if initializing
                    else elegy.States(optimizer_states=states.optimizer_states + 1),
                )

        model = Model()
        assert not hasattr(model.states, "optimizer_states")

        logs = model.train_on_batch(inputs=(np.array(1.0)), labels=(1.0,))
        assert logs["loss"] == 2
        assert model.states.optimizer_states == 1
        assert not hasattr(model.states, "net_params")

        model.eager = False

        logs = model.train_on_batch(inputs=(np.array(1.0)), labels=(1.0,))
        assert logs["loss"] == 2
        assert model.states.optimizer_states == 2
