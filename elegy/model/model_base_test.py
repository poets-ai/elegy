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

            step_states.metrics_states = 3

            if mode == "test":
                return step_states

            step_states.optimizer_states = 4

            return step_states

        def pred_step(
            self,
            parameters: tp.Any = None,
            x: tp.Any = (),
            pred_state: tp.Any = None,
        ) -> elegy.StepState:
            ...

        def test_step(
            self,
            parameters: tp.Any = None,
            x: tp.Any = (),
            y: tp.Any = None,
            pred_state: tp.Any = None,
            test_state: tp.Any = None,
            sample_weight: tp.Optional[jnp.ndarray] = None,
            class_weight: tp.Optional[jnp.ndarray] = None,
        ) -> elegy.StepState:
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