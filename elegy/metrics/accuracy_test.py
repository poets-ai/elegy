import unittest
import elegy


import jax.numpy as jnp


# import debugpy

# print("Waiting for debugger...")
# debugpy.listen(5679)
# debugpy.wait_for_client()


class AccuracyTest(unittest.TestCase):
    #
    def test_basic(self):

        accuracy = elegy.metrics.Accuracy()

        result = accuracy(
            y_true=jnp.array([1, 1, 1, 1]), y_pred=jnp.array([0, 1, 1, 1])
        )
        assert result == 0.75

        result = accuracy(
            y_true=jnp.array([1, 1, 1, 1]), y_pred=jnp.array([1, 0, 0, 0])
        )
        assert result == 0.5
