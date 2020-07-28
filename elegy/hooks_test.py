from unittest import TestCase
import elegy
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class TestHooks(TestCase):
    def test_context(self):
        @hk.to_module
        def my_module(x):
            w = elegy.get_parameter("w", [], jnp.float32, init=np.ones)
            elegy.get_state("s", [], jnp.float32, init=np.ones)
            elegy.add_metric("m", 1)
            elegy.add_loss("l", 1)
            elegy.set_state("s", w + 1)

            return elegy.nn.Linear(20)(x)

        states = None

        with elegy.context(states, rng=42) as ctx:
            y = my_module()(jnp.ones([3, 4]))

        states = ctx.collect()

        assert states
        assert y.shape == (3, 20)
        assert dict(states.params)["my_module"]["w"] == 1
        assert dict(states.state) == {"my_module": {"s": 1}}
        assert dict(states.metrics) == {"my_module/m": 1}
        assert dict(states.losses) == {"l_loss": 1}

        with elegy.context(states, rng=42) as ctx:
            y = my_module()(jnp.ones([3, 4]))

        states = ctx.collect()

        assert y.shape == (3, 20)
        assert states
        assert dict(states.params)["my_module"]["w"] == 1
        assert dict(states.state) == {"my_module": {"s": 2}}
        assert dict(states.metrics) == {"my_module/m": 1}
        assert dict(states.losses) == {"l_loss": 1}

