import unittest

import elegy
import jax
import jax.numpy as jnp
from elegy.generalized_module.generalized_module import generalize
import haiku
import numpy as np


class TestHaikuModule(unittest.TestCase):
    def test_basic(self):
        class M(haiku.Module):
            def __call__(self, x):

                n = haiku.get_state(
                    "n", shape=[], dtype=jnp.int32, init=lambda *args: np.array(0)
                )
                w = haiku.get_parameter("w", [], init=lambda *args: np.array(2.0))

                haiku.set_state("n", n + 1)

                return x * w

        def f(x, initializing, rng):
            return M()(x)

        gm = elegy.HaikuModule(f)
        rng = elegy.RNGSeq(42)

        y_true, params, states = gm.init(rng)(x=3.0, y=1, rng=None, initializing=True)

        assert y_true == 6
        assert params["m"]["w"] == 2
        assert states["m"]["n"] == 0

        params = haiku.data_structures.to_mutable_dict(params)
        params["m"]["w"] = np.array(10.0)
        y_true, params, states = gm.apply(params, states, training=True, rng=rng)(
            x=3.0, y=1, rng=None, initializing=True
        )

        assert y_true == 30
        assert params["m"]["w"] == 10
        assert states["m"]["n"] == 1

    def test_summaries(self):
        class ModuleC(haiku.Module):
            @elegy.haiku_summarize
            def __call__(self, x):
                c1 = haiku.get_parameter("c1", [5], jnp.int32, init=jnp.ones)
                c2 = haiku.get_state("c2", [6], jnp.int32, init=jnp.ones)

                x = jax.nn.relu(x)
                elegy.haiku_summary("relu", jax.nn.relu, x)

                return x

        class ModuleB(haiku.Module):
            @elegy.haiku_summarize
            def __call__(self, x):
                b1 = haiku.get_parameter("b1", [3], jnp.int32, init=jnp.ones)
                b2 = haiku.get_state("b2", [4], jnp.int32, init=jnp.ones)

                x = ModuleC()(x)

                x = jax.nn.relu(x)
                elegy.haiku_summary("relu", jax.nn.relu, x)

                return x

        class ModuleA(haiku.Module):
            @elegy.haiku_summarize
            def __call__(self, x):
                a1 = haiku.get_parameter("a1", [1], jnp.int32, init=jnp.ones)
                a2 = haiku.get_state("a2", [2], jnp.int32, init=jnp.ones)

                x = ModuleB()(x)

                x = jax.nn.relu(x)
                elegy.haiku_summary("relu", jax.nn.relu, x)

                return x

        def f(x):
            return ModuleA()(x)

        model = elegy.Model(elegy.HaikuModule(f))

        summary_text = model.summary(x=jnp.ones([10, 2]), depth=2, return_repr=True)
        assert summary_text is not None

        lines = summary_text.split("\n")

        assert "module_a/module_b" in lines[7]
        assert "ModuleB" in lines[7]
        assert "(10, 2)" in lines[7]
        assert "8" in lines[7]
        assert "32 B" in lines[7]
        assert "10" in lines[7]
        assert "40 B" in lines[7]

        assert "module_a/relu" in lines[9]
        assert "(10, 2)" in lines[9]

        assert "module_a" in lines[11]
        assert "ModuleA" in lines[11]
        assert "(10, 2)" in lines[11]
        assert "1" in lines[11]
        assert "4 B" in lines[11]
        assert "2" in lines[11]
        assert "8 B" in lines[11]

        assert "21" in lines[13]
        assert "84 B" in lines[13]

        assert "9" in lines[14]
        assert "36 B" in lines[14]

        assert "12" in lines[15]
        assert "48 B" in lines[15]
