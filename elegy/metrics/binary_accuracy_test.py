from unittest import TestCase

import jax.numpy as jnp
import tensorflow.keras as tfk
import numpy as np

import elegy


class BinaryCrossentropyTest(TestCase):
    #
    def test_example(self):
        y_true = np.array([[1], [1], [0], [0]])
        y_pred = np.array([[1], [1], [0], [0]])
        m = elegy.metrics.binary_accuracy(y_true, y_pred)
        assert m.shape == (4,)

        m = elegy.metrics.BinaryAccuracy()
        result = m.call_with_defaults()(
            y_true=np.array([[1], [1], [0], [0]]),
            y_pred=np.array([[0.98], [1], [0], [0.6]]),
        )
        assert result == 0.75

        m = elegy.metrics.BinaryAccuracy()
        result = m.call_with_defaults()(
            y_true=np.array([[1], [1], [0], [0]]),
            y_pred=np.array([[0.98], [1], [0], [0.6]]),
            sample_weight=np.array([1, 0, 0, 1]),
        )
        assert result == 0.5

    #
    def test_compatibility(self):

        y_true = (np.random.uniform(0, 1, size=(5, 6, 7)) > 0.5).astype(np.float32)
        y_pred = np.random.uniform(0, 1, size=(5, 6, 7))
        sample_weight = np.random.uniform(0, 1, size=(5, 6))

        assert np.allclose(
            tfk.metrics.BinaryAccuracy()(y_true, y_pred),
            elegy.metrics.BinaryAccuracy().call_with_defaults()(y_true, y_pred),
        )

        assert np.allclose(
            tfk.metrics.BinaryAccuracy(threshold=0.3)(y_true, y_pred),
            elegy.metrics.BinaryAccuracy(threshold=0.3).call_with_defaults()(
                y_true, y_pred
            ),
        )

        assert np.allclose(
            tfk.metrics.BinaryAccuracy(threshold=0.3)(
                y_true, y_pred, sample_weight=sample_weight
            ),
            elegy.metrics.BinaryAccuracy(threshold=0.3).call_with_defaults()(
                y_true, y_pred, sample_weight=sample_weight
            ),
        )

    def test_cumulative(self):

        tm = tfk.metrics.BinaryAccuracy(threshold=0.3)
        em = elegy.metrics.BinaryAccuracy(threshold=0.3)

        # 1st run
        y_true = (np.random.uniform(0, 1, size=(5, 6, 7)) > 0.5).astype(np.float32)
        y_pred = np.random.uniform(0, 1, size=(5, 6, 7))
        sample_weight = np.random.uniform(0, 1, size=(5, 6))

        assert np.allclose(
            tm(y_true, y_pred, sample_weight=sample_weight),
            em.call_with_defaults()(y_true, y_pred, sample_weight=sample_weight),
        )

        # 2nd run
        y_true = (np.random.uniform(0, 1, size=(5, 6, 7)) > 0.5).astype(np.float32)
        y_pred = np.random.uniform(0, 1, size=(5, 6, 7))
        sample_weight = np.random.uniform(0, 1, size=(5, 6))

        assert np.allclose(
            tm(y_true, y_pred, sample_weight=sample_weight),
            em.call_with_defaults()(y_true, y_pred, sample_weight=sample_weight),
        )
