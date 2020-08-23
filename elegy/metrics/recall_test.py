import unittest
import elegy

from elegy.testing_utils import transform_and_run
import jax.numpy as jnp


class RecallTest(unittest.TestCase):
    @transform_and_run
    def test_basic(self):

        recall = elegy.metrics.Recall()

        result = recall(y_true=jnp.array([0, 1, 1, 1]), y_pred=jnp.array([1, 0, 1, 1]))
        assert result == 0.6666667

        result = recall(y_true=jnp.array([1, 1, 1, 1]), y_pred=jnp.array([1, 0, 0, 0]))
        assert result == 0.42857143

        result = recall(
            y_true=jnp.array(
                [[0, 1, 1, 1], [0, 0, 1, 0], [0, 1, 1, 0], [1, 0, 1, 0], [0, 0, 1, 1]]
            ),
            y_pred=jnp.array(
                [[1, 0, 1, 1], [1, 0, 1, 1], [0, 0, 0, 1], [1, 1, 1, 1], [1, 0, 0, 1]]
            ),
        )
        assert result == 0.5294118

        result = recall(
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
        assert result == 0.5555556
