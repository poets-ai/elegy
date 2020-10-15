from unittest import TestCase

import jax.numpy as jnp
import tensorflow.keras as tfk
import numpy as np

import elegy
from elegy.testing_utils import transform_and_run


class F1Test(TestCase):
    @transform_and_run
    def test_basic(self):

        result = elegy.metrics.F1()(
            y_true=jnp.array([0, 1, 1, 1]), y_pred=jnp.array([1, 0, 1, 1])
        )
        assert np.allclose(result, 0.666667)  # 2 * (0.44445 / 1.33334)

        result = elegy.metrics.F1()(
            y_true=jnp.array([1, 1, 1, 1]), y_pred=jnp.array([1, 1, 0, 0])
        )
        assert np.allclose(result, 0.666667)  # 2 * (0.5 / 1.5)

    @transform_and_run
    def test_cummulative(self):
        em = elegy.metrics.F1(threshold=0.3)
        # 1st run
        y_true = jnp.array([0, 1, 1, 1])
        y_pred = jnp.array([1, 0, 1, 1])

        assert np.allclose(
            em(
                jnp.asarray(y_true),
                jnp.asarray(y_pred),
            ),
            0.666667,
        )

        # 2nd run
        y_true = jnp.array([1, 1, 1, 1])
        y_pred = jnp.array([1, 0, 0, 0])

        assert np.allclose(
            em(
                jnp.asarray(y_true),
                jnp.asarray(y_pred),
            ),
            0.5454545,
        )
