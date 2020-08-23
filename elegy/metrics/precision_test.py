import unittest
import elegy

from elegy.testing_utils import transform_and_run
import jax.numpy as jnp
import numpy as np


class PrecisionTest(unittest.TestCase):
    @transform_and_run
    def test_basic(self):

        precision = elegy.metrics.Precision(thresholds=0.6)

        result = precision(
            y_true=jnp.array([0, 1, 1, 1]), y_pred=jnp.array([1, 0, 1, 1])
        )
        assert np.isclose(result, 0.6666667)

        result = precision(
            y_true=jnp.array([1, 1, 1, 1]), y_pred=jnp.array([1, 1, 0, 0])
        )
        assert np.isclose(result, 0.8)

        result = precision(
            y_true=jnp.array(
                [[0, 1, 1, 1], [0, 0, 1, 0], [0, 1, 1, 0], [1, 0, 1, 0], [0, 0, 1, 1]]
            ),
            y_pred=jnp.array(
                [[1, 0, 1, 1], [1, 0, 1, 1], [0, 0, 0, 1], [1, 1, 1, 1], [1, 0, 0, 1]]
            ),
        )
        assert np.isclose(result, 0.5555556)

        result = precision(
            y_true=jnp.array(
                [[0, 1, 1, 1], [0, 0, 1, 0], [0, 1, 1, 0], [1, 0, 1, 0], [0, 0, 1, 1]]
            ),
            y_pred=jnp.array(
                [
                    [0.8, 0.0, 0.6, 0.7],
                    [0.6, 0.0, 0.6, 0.6],
                    [0.0, 0.0, 0.0, 0.7],
                    [0.7, 0.8, 0.6, 0.9],
                    [0.9, 0.0, 0.0, 0.8],
                ]
            ),
        )
        assert np.isclose(result, 0.5)
