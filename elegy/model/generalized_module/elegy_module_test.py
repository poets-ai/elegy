from unittest import TestCase

import elegy
from elegy.model.generalized_module.generalized_module import generalize
from flax import linen


class ElegyModuleTest(TestCase):
    def test_basic(self):
        class M(elegy.Module):
            def call(self, x):
                n = self.add_parameter("n", lambda: 0, trainable=False)
                w = self.add_parameter("w", lambda: 2.0)

                self.update_parameter("n", n + 1)

                key = elegy.next_key()

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
