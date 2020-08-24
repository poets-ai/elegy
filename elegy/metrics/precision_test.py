from unittest import TestCase

import jax.numpy as jnp
import tensorflow.keras as tfk
import numpy as np

import elegy
from elegy.testing_utils import transform_and_run


class PrecisionTest(TestCase):
    @transform_and_run
    def test_compatibility(self):


        a = tfk.metrics.Precision(thresholds=0.3)(
                np.array([0, 1, 1, 1]), np.array([0, 0, 1 , 1]), sample_weight=np.array([0, 1, 1, 0])
            )

        b = elegy.metrics.Precision(thresholds=0.3)(
                np.array([0, 1, 1, 1]), np.array([0, 0, 1, 1]), sample_weight=np.array([0, 1, 1, 1])
            )


        a = tfk.metrics.Precision(thresholds=0.3)(
                np.array([0, 1, 1, 1]), np.array([0, 0, 0 , 1]), sample_weight=np.array([0, 0, 1, 0])
            )

        b = elegy.metrics.Precision(thresholds=0.3)(
                np.array([0, 1, 1, 1]), np.array([0, 0, 0, 1]), sample_weight=np.array([0, 0, 1, 1])
            )

        y_true = (np.random.uniform(0, 1, size=(5, 6, 7)) > 0.5).astype(np.float32)
        y_pred = np.random.uniform(0, 1, size=(5, 6, 7))
        sample_weight = np.expand_dims(np.random.uniform(0, 1, size=(6, 7)), axis = 0)

        assert np.allclose(
            tfk.metrics.Precision()(y_true, y_pred),
            elegy.metrics.Precision()(y_true, y_pred),
        )

        assert np.allclose(
            tfk.metrics.Precision(thresholds=0.3)(y_true, y_pred),
            elegy.metrics.Precision(thresholds=0.3)(y_true, y_pred),
        )
            
        assert np.allclose(
            tfk.metrics.Precision(thresholds=0.3)(
                y_true, y_pred, sample_weight=sample_weight
            ),
            elegy.metrics.Precision(thresholds=0.3)(
                y_true, y_pred, sample_weight=sample_weight
            ),
        )

    @transform_and_run
    def test_cummulative(self):
        tm = tfk.metrics.Precision(thresholds=0.3)
        em = elegy.metrics.Precision(thresholds=0.3)

        # 1st run
        y_true = (np.random.uniform(0, 1, size=(5, 6, 7)) > 0.5).astype(np.float32)
        y_pred = np.random.uniform(0, 1, size=(5, 6, 7))
        sample_weight = np.expand_dims(np.random.uniform(0, 1, size=(6, 7)), axis=0)

        assert np.allclose(
            tm(y_true, y_pred, sample_weight=sample_weight),
            em(y_true, y_pred, sample_weight=sample_weight),
        )

        # 2nd run
        y_true = (np.random.uniform(0, 1, size=(5, 6, 7)) > 0.5).astype(np.float32)
        y_pred = np.random.uniform(0, 1, size=(5, 6, 7))
        sample_weight = np.expand_dims(np.random.uniform(0, 1, size=(6, 7)), axis=0)

        assert np.allclose(
            tm(y_true, y_pred, sample_weight=sample_weight),
            em(y_true, y_pred, sample_weight=sample_weight),
        )

test = PrecisionTest()
test.test_compatibility()
