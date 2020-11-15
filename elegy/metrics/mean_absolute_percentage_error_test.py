from unittest import TestCase
import elegy

import tensorflow.keras as tfk
import numpy as np
import jax.numpy as jnp


# import debugpy

# print("Waiting for debugger...")
# debugpy.listen(5679)
# debugpy.wait_for_client()


class MeanAbsolutePercentageErrorTest(TestCase):
    #
    def test_basic(self):

        y_true = np.random.random(size=(5, 6, 7))
        y_pred = np.random.random(size=(5, 6, 7))

        assert np.allclose(
            tfk.metrics.MeanAbsolutePercentageError()(y_true, y_pred),
            elegy.metrics.MeanAbsolutePercentageError()(
                jnp.asarray(y_true), jnp.asarray(y_pred)
            ),
        )

    #
    def test_cummulative(self):

        tm = tfk.metrics.MeanAbsolutePercentageError()
        em = elegy.metrics.MeanAbsolutePercentageError()

        # 1st run
        y_true = np.random.random(size=(5, 6, 7))
        y_pred = np.random.random(size=(5, 6, 7))

        assert np.allclose(
            tm(y_true, y_pred),
            em(jnp.asarray(y_true), jnp.asarray(y_pred)),
        )

        # 2nd run
        y_true = np.random.random(size=(5, 6, 7))
        y_pred = np.random.random(size=(5, 6, 7))

        assert np.allclose(
            tm(y_true, y_pred),
            em(jnp.asarray(y_true), jnp.asarray(y_pred)),
        )
