import typing as tp

import elegy
import jax.numpy as jnp
import numpy as np


def test_init():
    N = 0

    class Model(elegy.model.model_base.ModelBase):
        def init(self, mode: elegy.Mode) -> elegy.States:
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

        def pred_step(self) -> tp.Tuple[tp.Any, elegy.States]:
            ...

        def test_step(self) -> elegy.States:
            ...

        def train_step(self) -> elegy.States:
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
    class Model(elegy.model.model_base.ModelBase):
        def init(self, mode: elegy.Mode) -> elegy.States:
            return elegy.States()

        def pred_step(self, x) -> tp.Tuple[tp.Any, elegy.States]:
            return 1, elegy.States()

        def test_step(self) -> elegy.States:
            ...

        def train_step(self) -> elegy.States:
            ...

    model = Model()
    preds = model.predict_on_batch(x=(np.array(1.0)))

    assert preds == 1
