import elegy
from elegy.generalized_optimizer.generalized_optimizer import generalize_optimizer
import optax


def test_basic():

    w = 2.0
    grads = 1.5
    lr = 1.0
    rng = elegy.RNGSeq(42)

    go = generalize_optimizer(optax.sgd(lr))

    states = go.init(rng, w)
    w, states = go.apply(w, grads, states, rng)

    assert w == 0.5
