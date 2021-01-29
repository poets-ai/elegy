import jax.numpy as jnp

from unittest import TestCase
import jax

import elegy


class SequentialTest(TestCase):
    #
    def test_connects(self):

        y = elegy.nn.Sequential(
            lambda: [
                elegy.nn.Flatten(),
                elegy.nn.Linear(5),
                jax.nn.relu,
                elegy.nn.Linear(2),
            ]
        ).call_with_defaults(rng=elegy.RNGSeq(42))(jnp.ones([10, 3]))

        assert y.shape == (10, 2)

        y = elegy.nn.Sequential(
            lambda: [
                elegy.nn.Flatten(),
                elegy.nn.Linear(5),
                jax.nn.relu,
                elegy.nn.Linear(2),
            ]
        ).call_with_defaults(rng=elegy.RNGSeq(42), training=False)(jnp.ones([10, 3]))

        assert y.shape == (10, 2)

    def test_di(self):

        m = elegy.nn.Sequential(
            lambda: [
                elegy.nn.Flatten(),
                elegy.nn.Linear(2),
            ]
        )

        y = elegy.inject_dependencies(
            m.call_with_defaults(rng=elegy.RNGSeq(42), training=False), signature_f=m
        )(
            jnp.ones([5, 3]),
            a=1,
            b=2,
        )

        assert y.shape == (5, 2)
