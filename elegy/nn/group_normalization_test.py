import jax.numpy as jnp
from elegy.testing_utils import transform_and_run
from unittest import TestCase

import elegy


class GroupNormalizationTest(TestCase):
    def test_connects(self):
        elegy.nn.GroupNormalization(groups=5, create_scale=False, create_offset=False)(
            jnp.ones([3, 3]), training=True
        )

    def test_on_predict(self):
        class TestModule(elegy.Module):
            def call(self, x, training):
                return elegy.nn.GroupNormalization(groups=2)(x, training)

        model = elegy.model(module=TestModule())

        x = jnp.ones([3, 5, 5])

        y_pred = model.predict(x)
        logs = model.evaluate(x)
