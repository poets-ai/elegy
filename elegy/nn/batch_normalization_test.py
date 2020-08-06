import jax.numpy as jnp
from elegy.testing_utils import transform_and_run
from unittest import TestCase

import elegy


class BatchNormalizationTest(TestCase):
    def test_connects(self):
        elegy.nn.BatchNormalization()(jnp.ones([3, 3]), is_training=True)

    def test_on_predict(self):
        class TestModule(elegy.Module):
            def call(self, x, is_training):
                return elegy.nn.BatchNormalization()(x, is_training)

        model = elegy.Model(module=TestModule())

        x = jnp.ones([3, 5])

        y_pred = model.predict(x)
        logs = model.evaluate(x)
