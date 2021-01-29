from unittest import TestCase

import jax

import elegy
from elegy.model.generalized_module.generalized_module import generalize
import jax.numpy as jnp


class ElegyModuleTest(TestCase):
    def test_basic(self):
        class M(elegy.Module):
            def call(self, x):
                n = self.add_parameter("n", lambda: 0, trainable=False)
                w = self.add_parameter("w", lambda: 2.0)

                self.update_parameter("n", n + 1)

                key = self.next_key()

                return x * w

        gm = generalize(M())
        rng = elegy.RNGSeq(42)

        y_true, params, states = gm.init(rng)(x=3.0, y=1)

        assert y_true == 6
        assert params["w"] == 2
        assert states["states"]["n"] == 0

        params["w"] = 10.0
        y_true, params, states = gm.apply(params, states, rng)(x=3.0, y=1)

        assert y_true == 30
        assert params["w"] == 10
        assert states["states"]["n"] == 1

    def test_summaries(self):
        class ModuleC(elegy.Module):
            def call(self, x):
                c1 = self.add_parameter("c1", lambda: jnp.ones([5]))
                c2 = self.add_parameter("c2", lambda: jnp.ones([6]), trainable=False)

                x = jax.nn.relu(x)
                self.add_summary("relu", jax.nn.relu, x)

                return x

        class ModuleB(elegy.Module):
            def call(self, x):
                b1 = self.add_parameter("b1", lambda: jnp.ones([3]))
                b2 = self.add_parameter("b2", lambda: jnp.ones([4]), trainable=False)

                x = ModuleC()(x)

                x = jax.nn.relu(x)
                self.add_summary("relu", jax.nn.relu, x)

                return x

        class ModuleA(elegy.Module):
            def call(self, x):
                a1 = self.add_parameter("a1", lambda: jnp.ones([1]))
                a2 = self.add_parameter("a2", lambda: jnp.ones([2]), trainable=False)

                x = ModuleB()(x)

                x = jax.nn.relu(x)
                self.add_summary("relu", jax.nn.relu, x)

                return x

        model = elegy.Model(ModuleA())

        summary_text = model.summary(x=jnp.ones([10, 2]), depth=10)

        lines = summary_text.split("\n")

        assert "module_b/module_c/relu" in lines[7]
        assert "(10, 2)" in lines[7]

        assert "module_b/module_c" in lines[9]
        assert "ModuleC" in lines[9]
        assert "(10, 2)" in lines[9]
        assert "5" in lines[9]
        assert "20 B" in lines[9]
        assert "6" in lines[9]
        assert "24 B" in lines[9]

        assert "module_b/relu" in lines[11]
        assert "(10, 2)" in lines[11]

        assert "module_b" in lines[13]
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
