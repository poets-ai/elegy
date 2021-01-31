from unittest import TestCase

import jax.numpy as jnp
import tensorflow.keras as tfk
import numpy as np

import elegy


class RecallTest(TestCase):
    def test_basic(self):

        y_true = (np.random.uniform(0, 1, size=(5, 6, 7)) > 0.5).astype(np.float32)
        y_pred = np.random.uniform(0, 1, size=(5, 6, 7))
        sample_weight = np.expand_dims(
            (np.random.uniform(0, 1, size=(6, 7)) > 0.5).astype(int), axis=0
        )

        assert np.allclose(
            tfk.metrics.Recall()(y_true, y_pred),
            elegy.metrics.Recall().call_with_defaults()(
                jnp.asarray(y_true), jnp.asarray(y_pred)
            ),
        )

        assert np.allclose(
            tfk.metrics.Recall(thresholds=0.3)(y_true, y_pred),
            elegy.metrics.Recall(threshold=0.3).call_with_defaults()(
                jnp.asarray(y_true), jnp.asarray(y_pred)
            ),
        )

        assert np.allclose(
            tfk.metrics.Recall(thresholds=0.3)(
                y_true, y_pred, sample_weight=sample_weight
            ),
            elegy.metrics.Recall(threshold=0.3).call_with_defaults()(
                jnp.asarray(y_true),
                jnp.asarray(y_pred),
                sample_weight=jnp.asarray(sample_weight),
            ),
        )

        float_sample_weight = np.random.uniform(0, 1, size=(6, 7))[np.newaxis]
        assert np.allclose(
            tfk.metrics.Recall(thresholds=0.3)(
                y_true, y_pred, sample_weight=float_sample_weight
            ),
            elegy.metrics.Recall(threshold=0.3).call_with_defaults()(
                y_true, y_pred, sample_weight=float_sample_weight
            ),
        )

    def test_cumulative(self):
        tm = tfk.metrics.Recall(thresholds=0.3)
        em = elegy.metrics.Recall(threshold=0.3)

        # 1st run
        y_true = (np.random.uniform(0, 1, size=(5, 6, 7)) > 0.5).astype(np.float32)
        y_pred = np.random.uniform(0, 1, size=(5, 6, 7))
        sample_weight = np.expand_dims(
            (np.random.uniform(0, 1, size=(6, 7)) > 0.5).astype(int), axis=0
        )

        assert np.allclose(
            tm(y_true, y_pred, sample_weight=sample_weight),
            em.call_with_defaults()(
                jnp.asarray(y_true),
                jnp.asarray(y_pred),
                sample_weight=jnp.asarray(sample_weight),
            ),
        )

        # 2nd run
        y_true = (np.random.uniform(0, 1, size=(5, 6, 7)) > 0.5).astype(np.float32)
        y_pred = np.random.uniform(0, 1, size=(5, 6, 7))
        sample_weight = np.expand_dims(
            (np.random.uniform(0, 1, size=(6, 7)) > 0.5).astype(int), axis=0
        )

        assert np.allclose(
            tm(y_true, y_pred, sample_weight=sample_weight),
            em.call_with_defaults()(
                jnp.asarray(y_true),
                jnp.asarray(y_pred),
                sample_weight=jnp.asarray(sample_weight),
            ),
        )
