import jax.numpy as jnp

from unittest import TestCase
import jax

import elegy


class SequentialTest(TestCase):
    #
    def test_connects(self):

        with elegy.update_context(rng=elegy.RNGSeq(42)):
            y = elegy.nn.Sequential(
                lambda: [
                    elegy.nn.Flatten(),
                    elegy.nn.Linear(5),
                    jax.nn.relu,
                    elegy.nn.Linear(2),
                ]
            )(jnp.ones([10, 3]))

            assert y.shape == (10, 2)

        with elegy.update_context(rng=elegy.RNGSeq(42), training=False):
            y = elegy.nn.Sequential(
                lambda: [
                    elegy.nn.Flatten(),
                    elegy.nn.Linear(5),
                    jax.nn.relu,
                    elegy.nn.Linear(2),
                ]
            )(jnp.ones([10, 3]))

            assert y.shape == (10, 2)
