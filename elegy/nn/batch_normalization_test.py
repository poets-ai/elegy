import jax.numpy as jnp

from unittest import TestCase

import elegy


class BatchNormalizationTest(TestCase):
    def test_connects(self):
        elegy.nn.BatchNormalization().call_with_defaults()(
            jnp.ones([3, 3]),
            training=True,
        )

    def test_on_predict(self):
        class TestModule(elegy.Module):
            def call(self, x, training):
                return elegy.nn.BatchNormalization()(x, training)

        model = elegy.Model(module=TestModule())

        x = jnp.ones([3, 5])

        y_pred = model.predict(x)
        logs = model.evaluate(x)
