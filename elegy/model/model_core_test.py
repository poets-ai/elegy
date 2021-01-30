import typing as tp
import unittest

import elegy
import jax.numpy as jnp
import numpy as np


class ModelCoreTest(unittest.TestCase):
    def test_init(self):
        N = 0

        class Model(elegy.model.model_core.ModelCore):
            def pred_step(self):
                nonlocal N
                N = N + 1

                return elegy.PredStep.simple(
                    y_pred=None,
                    states=elegy.States(net_params=1, net_states=2),
                )

            def test_step(self):
                _, states, _, _, _ = self.pred_step()
                return elegy.TestStep(0, {}, states.update(metrics_states=3))

            def train_step(self):
                _, logs, states = self.test_step()
                return elegy.TrainStep(logs, states.update(optimizer_states=4))

        model = Model()

        assert N == 0
        assert model.states.net_params != 1
        assert model.states.net_states != 2
        assert model.states.metrics_states != 3
        assert model.states.optimizer_states != 4

        model.maybe_initialize(elegy.types.Mode.pred)

        assert N == 1
        assert model.states.net_params == 1
        assert model.states.net_states == 2

        model.maybe_initialize(elegy.types.Mode.pred)

        assert N == 1

        model.maybe_initialize(elegy.types.Mode.test)

        assert N == 2
        assert model.states.metrics_states == 3

        model.maybe_initialize(elegy.types.Mode.test)

        assert N == 2

        model.maybe_initialize(elegy.types.Mode.train)

        assert N == 3
        assert model.states.optimizer_states == 4

        model.maybe_initialize(elegy.types.Mode.train)

        assert N == 3

    def test_pred_step(self):
        class Model(elegy.model.model_core.ModelCore):
            def pred_step(self, x, net_states, initializing):
                if initializing:
                    states = elegy.States(net_states=0)
                else:
                    states = elegy.States(net_states=net_states + 1)

                return elegy.PredStep.simple(
                    y_pred=1,
                    states=states,
                )

            def test_step(self):
                ...

            def train_step(self):
                ...

        model = Model()
        assert isinstance(model.states.net_states, elegy.types.Uninitialized)

        preds = model.predict_on_batch(x=(np.array(1.0)))
        assert preds == 1
        assert model.states.net_states == 1
        assert isinstance(model.states.net_params, elegy.types.Uninitialized)

        model.run_eagerly = False

        preds = model.predict_on_batch(x=(np.array(1.0)))
        assert preds == 1
        assert model.states.net_states == 2

    def test_test_step(self):
        class Model(elegy.model.model_core.ModelCore):
            def test_step(self, metrics_states, initializing):
                return elegy.TestStep(
                    loss=0.1,
                    logs=dict(loss=1.0),
                    states=elegy.States(metrics_states=0)
                    if initializing
                    else elegy.States(metrics_states=metrics_states + 1),
                )

            def train_step(self):
                ...

        model = Model()
        assert isinstance(model.states.metrics_states, elegy.types.Uninitialized)

        logs = model.test_on_batch(x=(np.array(1.0)), y=(1.0,))
        assert logs["loss"] == 1
        assert model.states.metrics_states == 1
        assert isinstance(model.states.net_params, elegy.types.Uninitialized)

        model.run_eagerly = False

        logs = model.test_on_batch(x=(np.array(1.0)), y=(1.0,))
        assert logs["loss"] == 1
        assert model.states.metrics_states == 2

    def test_train_step(self):
        class Model(elegy.model.model_core.ModelCore):
            def train_step(self, optimizer_states, initializing):
                return elegy.TrainStep(
                    logs=dict(loss=2.0),
                    states=elegy.States(optimizer_states=0)
                    if initializing
                    else elegy.States(optimizer_states=optimizer_states + 1),
                )

        model = Model()
        assert isinstance(model.states.optimizer_states, elegy.types.Uninitialized)

        logs = model.train_on_batch(x=(np.array(1.0)), y=(1.0,))
        assert logs["loss"] == 2
        assert model.states.optimizer_states == 1
        assert isinstance(model.states.net_params, elegy.types.Uninitialized)

        model.run_eagerly = False

        logs = model.train_on_batch(x=(np.array(1.0)), y=(1.0,))
        assert logs["loss"] == 2
        assert model.states.optimizer_states == 2
