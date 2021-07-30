from unittest import TestCase

import jax

import elegy
from elegy.generalized_module.generalized_module import generalize
import jax.numpy as jnp


class ElegyModuleTest(TestCase):
    def test_basic(self):
        def M(x, initializing, params, states):
            if initializing:
                params = {"w": 2.0}
                states = {"n": 0}
            else:
                states["n"] += 1

            return elegy.OutputStates(x * params["w"], params, states)

        gm = generalize(M)
        rng = elegy.RNGSeq(42)

        y_true, params, states = gm.init(rng)(
            x=3.0, y=1, initializing=True, params=None, states=None
        )

        assert y_true == 6
        assert params["w"] == 2
        assert states["n"] == 0

        params["w"] = 10.0
        y_true, params, states = gm.apply(params, states, training=True, rng=rng)(
            x=3.0,
            y=1,
            initializing=False,
            params=params,
            states=states,
        )

        assert y_true == 30
        assert params["w"] == 10
        assert states["n"] == 1

    def test_summaries(self):
        def M(x, initializing, states):
            if initializing:
                net_params = {"w": 2.0}
                net_states = {"n": 0}
            else:
                net_params = states.net_params
                net_states = states.net_states

                net_states["n"] += 1

            return elegy.OutputStates(x * net_params["w"], net_params, net_states)

        model = elegy.Model(M)

        x = jnp.ones([10, 2])

        summary_text = model.summary(x, depth=1, return_repr=True)
        assert summary_text is not None

        lines = summary_text.split("\n")

        assert "(10, 2)" in lines[3]
        assert "(10, 2)" in lines[5]

        assert "1" in lines[12]
        assert "4 B" in lines[12]

        assert "2" in lines[15]
        assert "8 B" in lines[15]
