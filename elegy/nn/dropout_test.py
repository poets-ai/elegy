from elegy.testing_utils import transform_and_run
from elegy import utils

import jax.numpy as jnp
from unittest import TestCase

import elegy


class DropoutTest(TestCase):
    @transform_and_run
    def test_dropout_connects(self):
        elegy.nn.Dropout(0.25)(jnp.ones([3, 3]), is_training=True)

    def test_on_predict(self):
        class TestModule(elegy.Module):
            def call(self, x, is_training):
                return elegy.nn.Dropout(0.5)(x, is_training)

        model = elegy.Model(module=TestModule.defer())

        x = jnp.ones([3, 5])

        y_pred = model.predict(x)
        logs = model.evaluate(x)

        assert jnp.all(y_pred == x)
