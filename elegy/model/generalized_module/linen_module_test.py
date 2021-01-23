import unittest

import elegy
import jax
import jax.numpy as jnp
from elegy.model.generalized_module.generalized_module import generalize
from flax import linen


class TestLinenModule(unittest.TestCase):
    def test_basic(self):
        class M(linen.Module):
            @linen.compact
            def __call__(self, x):

                initialized = self.has_variable("batch_stats", "n")

                vn = self.variable("batch_stats", "n", lambda: 0)

                w = self.param("w", lambda key: 2.0)

                if initialized:
                    vn.value += 1

                return x * w

        gm = generalize(M())
        rng = elegy.RNGSeq(42)

        y_true, params, states = gm.init(rng)(x=3.0, y=1)

        assert y_true == 6
        assert params["w"] == 2
        assert states["batch_stats"]["n"] == 0

        params = params.copy(dict(w=10.0))
        y_true, params, states = gm.apply(params, states, rng)(x=3.0, y=1)

        assert y_true == 30
        assert params["w"] == 10
        assert states["batch_stats"]["n"] == 1

    def test_summaries(self):
        class ModuleC(linen.Module):
            @linen.compact
            @elegy.flax_summarize
            def __call__(self, x):
                c1 = self.param("c1", lambda _: jnp.ones([5]))
                c2 = self.variable("states", "c2", lambda: jnp.ones([6]))

                x = jax.nn.relu(x)
                elegy.flax_summary(self, "relu", jax.nn.relu, x)

                return x

        class ModuleB(linen.Module):
            @linen.compact
            @elegy.flax_summarize
            def __call__(self, x):
                b1 = self.param("b1", lambda _: jnp.ones([3]))
                b2 = self.variable("states", "b2", lambda: jnp.ones([4]))

                x = ModuleC()(x)

                x = jax.nn.relu(x)
                elegy.flax_summary(self, "relu", jax.nn.relu, x)

                return x

        class ModuleA(linen.Module):
            @linen.compact
            @elegy.flax_summarize
            def __call__(self, x):
                a1 = self.param("a1", lambda _: jnp.ones([1]))
                a2 = self.variable("states", "a2", lambda: jnp.ones([2]))

                x = ModuleB()(x)

                x = jax.nn.relu(x)
                elegy.flax_summary(self, "relu", jax.nn.relu, x)

                return x

        model = elegy.Model(ModuleA())

        summary_text = model.summary(x=jnp.ones([10, 2]), depth=10)

        lines = summary_text.split("\n")

        assert "ModuleB_0/ModuleC_0/relu" in lines[7]
        assert "(10, 2)" in lines[7]

        assert "ModuleB_0/ModuleC_0" in lines[9]
        assert "ModuleC" in lines[9]
        assert "(10, 2)" in lines[9]
        assert "5" in lines[9]
        assert "20 B" in lines[9]
        assert "6" in lines[9]
        assert "24 B" in lines[9]

        assert "ModuleB_0/relu" in lines[11]
        assert "(10, 2)" in lines[11]

        assert "ModuleB_0" in lines[13]
        assert "ModuleB" in lines[13]
        assert "(10, 2)" in lines[13]
        assert "3" in lines[13]
        assert "12 B" in lines[13]
        assert "4" in lines[13]
        assert "16 B" in lines[13]

        assert "relu" in lines[15]
        assert "(10, 2)" in lines[15]

        assert "*" in lines[17]
        assert "ModuleA" in lines[17]
        assert "(10, 2)" in lines[17]
        assert "1" in lines[17]
        assert "4 B" in lines[17]
        assert "2" in lines[17]
        assert "8 B" in lines[17]

        assert "21" in lines[19]
        assert "84 B" in lines[19]

        assert "9" in lines[20]
        assert "36 B" in lines[20]

        assert "12" in lines[21]
        assert "48 B" in lines[21]
