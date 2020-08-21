import unittest
import elegy

from elegy.testing_utils import transform_and_run
import jax.numpy as jnp


class PrecisionTest(unittest.TestCase):
    @transform_and_run
    def test_basic(self):

        precision = elegy.metrics.Precision()
        
        result = precision(
            y_true=jnp.array([0, 1, 1, 1]), y_pred=jnp.array([1, 0, 1, 1])
        )
        assert result == 0.6666667

        result = precision(
            y_true=jnp.array([1, 1, 1, 1]), y_pred=jnp.array([1, 1, 0, 0])
        )
        assert result == 0.8
