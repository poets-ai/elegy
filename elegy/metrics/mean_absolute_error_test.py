from unittest import TestCase

import jax.numpy as jnp
import tensorflow.keras as tfk
import numpy as np
import elegy


class MAETest(TestCase):
    def test_basic(self):

        y_true = jnp.array([1, 1, 1, 1])
        y_pred = jnp.array([0, 1, 1, 1])

        assert np.allclose(
            elegy.metrics.MeanAbsoluteError().call_with_defaults()(y_true, y_pred),
            tfk.metrics.MeanAbsoluteError()(y_true, y_pred),
        )

        y_true = jnp.array([1, 1, 1, 1])
        y_pred = jnp.array([1, 0, 0, 0])

        assert np.allclose(
            elegy.metrics.MeanAbsoluteError().call_with_defaults()(y_true, y_pred),
            tfk.metrics.MeanAbsoluteError()(y_true, y_pred),
        )

        y_true = (np.random.uniform(0, 1, size=(5, 6, 7)) > 0.5).astype(np.int32)
        y_pred = np.random.uniform(0, 1, size=(5, 6, 7))

        assert np.allclose(
            tfk.metrics.MeanAbsoluteError()(y_true, y_pred),
            elegy.metrics.MeanAbsoluteError().call_with_defaults()(y_true, y_pred),
        )
