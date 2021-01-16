import typing as tp

import elegy
import jax.numpy as jnp
import numpy as np


def test_init():
    N = 0

    class Model(elegy.model.model_core.ModelCore):
        def init(self, mode: elegy.Mode):
            nonlocal N
            N = N + 1

            step_states = elegy.States(net_params=1, net_states=2)

            if mode == "pred":
                return step_states

            step_states = step_states.update(metrics_states=3)

            if mode == "test":
                return step_states

            step_states = step_states.update(optimizer_states=4)

            return step_states

        def pred_step(self):
            ...

        def test_step(self):
            ...

        def train_step(self):
            ...

    model = Model()

    assert N == 0
    assert model.states.net_params != 1
    assert model.states.net_states != 2
    assert model.states.metrics_states != 3
    assert model.states.optimizer_states != 4

    model.maybe_initialize(mode=elegy.Mode.pred)

    assert N == 1
    assert model.states.net_params == 1
    assert model.states.net_states == 2

    model.maybe_initialize(mode=elegy.Mode.pred)

    assert N == 1

    model.maybe_initialize(mode=elegy.Mode.test)

    assert N == 2
    assert model.states.metrics_states == 3

    model.maybe_initialize(mode=elegy.Mode.test)

    assert N == 2

    model.maybe_initialize(mode=elegy.Mode.train)

    assert N == 3
    assert model.states.optimizer_states == 4

    model.maybe_initialize(mode=elegy.Mode.train)

    assert N == 3


def test_pred_step():
    class Model(elegy.model.model_core.ModelCore):
        def init(self, mode: elegy.Mode):
            return elegy.States(net_states=0)

        def pred_step(self, x, net_states):
            return elegy.Prediction(
                pred=1,
                states=elegy.States(net_states=net_states + 1),
            )

        def test_step(self):
            ...

        def train_step(self):
            ...

    model = Model()
    assert isinstance(model.states.net_states, elegy.utils.Uninitialized)

    preds = model.predict_on_batch(x=(np.array(1.0)))
    assert preds == 1
    assert model.states.net_states == 1
    assert isinstance(model.states.net_params, elegy.utils.Uninitialized)

    model.run_eagerly = False

    preds = model.predict_on_batch(x=(np.array(1.0)))
    assert preds == 1
    assert model.states.net_states == 2


def test_test_step():
    class Model(elegy.model.model_core.ModelCore):
        def init(self, mode: elegy.Mode):
            return elegy.States(metrics_states=0)

        def pred_step(self, x):
            return 1, elegy.States()

        def test_step(self, metrics_states):
            return elegy.Evaluation(
                loss=0.1,
                logs=dict(loss=1.0),
                states=elegy.States(metrics_states=metrics_states + 1),
            )

        def train_step(self):
            ...

    model = Model()
    assert isinstance(model.states.metrics_states, elegy.utils.Uninitialized)

    logs = model.test_on_batch(x=(np.array(1.0)), y=(1.0,))
    assert logs["loss"] == 1
    assert model.states.metrics_states == 1
    assert isinstance(model.states.net_params, elegy.utils.Uninitialized)

    model.run_eagerly = False

    logs = model.test_on_batch(x=(np.array(1.0)), y=(1.0,))
    assert logs["loss"] == 1
    assert model.states.metrics_states == 2


def test_train_step():
    class Model(elegy.model.model_core.ModelCore):
        def init(self, mode: elegy.Mode):
            return elegy.States(optimizer_states=0)

        def pred_step(self, x):
            return 1, elegy.States()

        def test_step(self):
            return dict(loss=1.0), elegy.States()

        def train_step(self, optimizer_states):
            return elegy.Training(
                logs=dict(loss=2.0),
                states=elegy.States(optimizer_states=optimizer_states + 1),
            )

    model = Model()
    assert isinstance(model.states.optimizer_states, elegy.utils.Uninitialized)

    logs = model.train_on_batch(x=(np.array(1.0)), y=(1.0,))
    assert logs["loss"] == 2
    assert model.states.optimizer_states == 1
    assert isinstance(model.states.net_params, elegy.utils.Uninitialized)

    model.run_eagerly = False

    logs = model.train_on_batch(x=(np.array(1.0)), y=(1.0,))
    assert logs["loss"] == 2
    assert model.states.optimizer_states == 2