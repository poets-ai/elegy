import typing as tp
import unittest

import elegy
import jax.numpy as jnp
import numpy as np


class ModelCoreTest(unittest.TestCase):
    def test_init(self):
        N = 0

        class Model(elegy.model.model_core.ModelCore):
            def states_step(self):
                return elegy.States(
                    net_params=None,
                    net_states=None,
                    metrics_states=None,
                    optimizer_states=None,
                )

            def init_step(self, states):
                states = states.maybe_update(**self.states_step())
                _, states = self.train_step(states)

                return states

            def pred_step(self, states):
                nonlocal N
                N = N + 1

                return elegy.PredStep(
                    y_pred=None,
                    states=states.update(net_params=1, net_states=2),
                )

            def test_step(self, states):
                _, states = self.pred_step(states)
                return elegy.TestStep(0, {}, states.update(metrics_states=3))

            def train_step(self, states):
                _, logs, states = self.test_step(states)
                return elegy.TrainStep(logs, states.update(optimizer_states=4))

        model = Model()

        assert N == 0
        assert not hasattr(model.states, "net_params")
        assert not hasattr(model.states, "net_states")
        assert not hasattr(model.states, "metrics_states")
        assert not hasattr(model.states, "optimizer_states")

        model.init_on_batch()

        assert N == 1
        assert model.states.net_params == 1
        assert model.states.net_states == 2
        assert model.states.metrics_states == 3
        assert model.states.optimizer_states == 4

        model.init_on_batch()
        assert N == 1

    def test_pred_step(self):
        class Model(elegy.model.model_core.ModelCore):
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
                    states=states,
                )

        model = Model()
        assert not hasattr(model.states, "net_states")

        model.init_on_batch(x=(np.array(1.0)))
        preds = model.predict_on_batch(x=(np.array(1.0)))
        assert preds == 1
        assert model.states.net_states == 1
        assert not hasattr(model.states, "net_params")

        model.run_eagerly = False

        preds = model.predict_on_batch(x=(np.array(1.0)))
        assert preds == 1
        assert model.states.net_states == 2

    def test_test_step(self):
        class Model(elegy.model.model_core.ModelCore):
            def init_step(self, states):
                _, _, states = self.test_step(states, True)
                return states

            def test_step(self, states, initializing):
                return elegy.TestStep(
                    loss=0.1,
                    logs=dict(loss=1.0),
                    states=elegy.States(metrics_states=0)
                    if initializing
                    else elegy.States(metrics_states=states.metrics_states + 1),
                )

            def train_step(self):
                ...

        model = Model()
        assert not hasattr(model.states, "metrics_states")

        logs = model.test_on_batch(x=(np.array(1.0)), y=(1.0,))
        assert logs["loss"] == 1
        assert model.states.metrics_states == 1
        assert not hasattr(model.states, "net_params")

        model.run_eagerly = False

        logs = model.test_on_batch(x=(np.array(1.0)), y=(1.0,))
        assert logs["loss"] == 1
        assert model.states.metrics_states == 2

    def test_train_step(self):
        class Model(elegy.model.model_core.ModelCore):
            def init_step(self, states):
                _, states = self.train_step(states, True)
                return states

            def train_step(self, states, initializing):
                return elegy.TrainStep(
                    logs=dict(loss=2.0),
                    states=elegy.States(optimizer_states=0)
                    if initializing
                    else elegy.States(optimizer_states=states.optimizer_states + 1),
                )

        model = Model()
        assert not hasattr(model.states, "optimizer_states")

        logs = model.train_on_batch(x=(np.array(1.0)), y=(1.0,))
        assert logs["loss"] == 2
        assert model.states.optimizer_states == 1
        assert not hasattr(model.states, "net_params")

        model.run_eagerly = False

        logs = model.train_on_batch(x=(np.array(1.0)), y=(1.0,))
        assert logs["loss"] == 2
        assert model.states.optimizer_states == 2
