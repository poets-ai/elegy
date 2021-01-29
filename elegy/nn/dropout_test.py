from elegy import utils

import jax.numpy as jnp
from unittest import TestCase

import elegy


class DropoutTest(TestCase):
    def test_dropout_connects(self):
        elegy.nn.Dropout(0.25).call_with_defaults(rng=elegy.RNGSeq(42))(
            jnp.ones([3, 3]), training=True
        )

    def test_on_predict(self):
        class TestModule(elegy.Module):
            def call(self, x, training):
                return elegy.nn.Dropout(0.5)(x, training)

        model = elegy.Model(TestModule())

        x = jnp.ones([3, 5])

        y_pred = model.predict(x)
        logs = model.evaluate(x)

        assert jnp.all(y_pred == x)
