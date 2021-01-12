import typing as tp

import elegy
import jax.numpy as jnp
import numpy as np


def test_init():
    N = 0

    class Model(elegy.model.model_base.ModelBase):
        def init(self, mode: elegy.Mode) -> elegy.StepState:
            nonlocal N
            N = N + 1

            step_states = elegy.StepState(parameters=1, states=2)

            if mode == "pred":
                return step_states

            step_states = step_states.update(metrics_states=3)

            if mode == "test":
                return step_states

            step_states = step_states.update(optimizer_states=4)

            return step_states

        def pred_step(self) -> tp.Tuple[tp.Any, elegy.StepState]:
            ...

        def test_step(self) -> elegy.StepState:
            ...

        def train_step(self) -> elegy.StepState:
            ...

    model = Model()

    assert N == 0
    assert model.parameters != 1
    assert model.states != 2
    assert model.metrics_states != 3
    assert model.optimizer_states != 4

    model.maybe_initialize(mode=elegy.Mode.pred)

    assert N == 1
    assert model.parameters == 1
    assert model.states == 2

    model.maybe_initialize(mode=elegy.Mode.pred)

    assert N == 1

    model.maybe_initialize(mode=elegy.Mode.test)

    assert N == 2
    assert model.metrics_states == 3

    model.maybe_initialize(mode=elegy.Mode.test)

    assert N == 2

    model.maybe_initialize(mode=elegy.Mode.train)

    assert N == 3
    assert model.optimizer_states == 4

    model.maybe_initialize(mode=elegy.Mode.train)

    assert N == 3


def test_pred_step():
    class Model(elegy.model.model_base.ModelBase):
        def init(self, mode: elegy.Mode) -> elegy.StepState:
            return elegy.StepState()

        def pred_step(self, x) -> tp.Tuple[tp.Any, elegy.StepState]:
            return 1, self.step_states

        def test_step(self) -> elegy.StepState:
            ...

        def train_step(self) -> elegy.StepState:
            ...

    model = Model()
    preds = model.predict_on_batch(x=(np.array(1.0)))

    assert preds == 1