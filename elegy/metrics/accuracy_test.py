import unittest
import elegy


import jax.numpy as jnp


class AccuracyTest(unittest.TestCase):
    #
    def test_basic(self):

        accuracy = elegy.metrics.Accuracy()

        result = accuracy.call_with_defaults()(
            y_true=jnp.array([1, 1, 1, 1]), y_pred=jnp.array([0, 1, 1, 1])
        )
        assert result == 0.75

        result = accuracy.call_with_defaults_jit()(
            jnp.array([1, 1, 1, 1]), jnp.array([1, 0, 0, 0])
        )
        assert result == 0.5
