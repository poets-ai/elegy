import elegy
from elegy.model.generalized_module.generalized_module import generalize
from flax import linen


def test_basic():
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

    y_true, params, states = gm.init(rng)(x=3.0)

    assert y_true == 6
    assert params["w"] == 2
    assert states["batch_stats"]["n"] == 0

    params = params.copy(dict(w=10.0))
    y_true, params, states = gm.apply(params, states, rng)(x=3.0)

    assert y_true == 30
    assert params["w"] == 10
    assert states["batch_stats"]["n"] == 1