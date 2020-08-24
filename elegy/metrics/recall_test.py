import unittest
import elegy

from elegy.testing_utils import transform_and_run
import jax.numpy as jnp
import numpy as np


class RecallTest(unittest.TestCase):
    @transform_and_run
    def test_basic(self):

        y_true = (np.random.uniform(0, 1, size=(5, 6, 7)) > 0.5).astype(np.float32)
        y_pred = np.random.uniform(0, 1, size=(5, 6, 7))
        sample_weight = np.expand_dims(np.random.uniform(0, 1, size=(6, 7)), axis = 0)

        assert np.allclose(
            tfk.metrics.Recall()(y_true, y_pred),
            elegy.metrics.Recall()(y_true, y_pred),
        )

        assert np.allclose(
            tfk.metrics.Recall(thresholds=0.3)(y_true, y_pred),
            elegy.metrics.Recall(thresholds=0.3)(y_true, y_pred),
        )
            
        assert np.allclose(
            tfk.metrics.Recall(thresholds=0.3)(
                y_true, y_pred, sample_weight=sample_weight
            ),
            elegy.metrics.Recall(thresholds=0.3)(
                y_true, y_pred, sample_weight=sample_weight
            ),
        )

    def test_cummulative(self):
        tm = tfk.metrics.Recall(thresholds=0.3)
        em = elegy.metrics.Recall(thresholds=0.3)

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
