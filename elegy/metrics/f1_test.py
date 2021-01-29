from unittest import TestCase

import jax.numpy as jnp
import tensorflow_addons as tfa
import numpy as np
import elegy


class F1Test(TestCase):
    #
    def test_basic(self):

        y_true = jnp.array([0, 1, 1, 1])
        y_pred = jnp.array([1, 0, 1, 1])

        assert np.allclose(
            elegy.metrics.F1().call_with_defaults()(y_true, y_pred),
            tfa.metrics.F1Score(2, average="micro", threshold=0.5)(y_true, y_pred),
        )  # 2 * (0.44445 / 1.33334)

        y_true = jnp.array([1, 1, 1, 1])
        y_pred = jnp.array([1, 1, 0, 0])

        assert np.allclose(
            elegy.metrics.F1().call_with_defaults()(y_true, y_pred),
            tfa.metrics.F1Score(2, average="micro", threshold=0.5)(y_true, y_pred),
        )  # 2 * (0.5 / 1.5)

        y_true = (np.random.uniform(0, 1, size=(5, 6, 7)) > 0.5).astype(np.float32)
        y_pred = np.random.uniform(0, 1, size=(5, 6, 7))
        sample_weight = np.expand_dims(
            (np.random.uniform(0, 1, size=(6, 7)) > 0.5).astype(np.float32), axis=0
        )

        assert np.allclose(
            tfa.metrics.F1Score(2, average="micro", threshold=0.3)(y_true, y_pred),
            elegy.metrics.F1(threshold=0.3).call_with_defaults()(
                jnp.asarray(y_true), jnp.asarray(y_pred)
            ),
        )

        assert np.allclose(
            tfa.metrics.F1Score(2, average="micro", threshold=0.3)(
                y_true, y_pred, sample_weight=sample_weight
            ),
            elegy.metrics.F1(threshold=0.3).call_with_defaults()(
                jnp.asarray(y_true), jnp.asarray(y_pred), sample_weight=sample_weight
            ),
        )

    #
    def test_cumulative(self):
        em = elegy.metrics.F1(threshold=0.3)
        tm = tfa.metrics.F1Score(2, average="micro", threshold=0.3)

        # 1st run
        y_true = (np.random.uniform(0, 1, size=(5, 6, 7)) > 0.5).astype(np.float32)
        y_pred = np.random.uniform(0, 1, size=(5, 6, 7))
        sample_weight = np.expand_dims(
            (np.random.uniform(0, 1, size=(6, 7)) > 0.5).astype(np.float32), axis=0
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
            (np.random.uniform(0, 1, size=(6, 7)) > 0.5).astype(np.float32), axis=0
        )

        assert np.allclose(
            tm(y_true, y_pred, sample_weight=sample_weight),
            em.call_with_defaults()(
                jnp.asarray(y_true),
                jnp.asarray(y_pred),
                sample_weight=jnp.asarray(sample_weight),
            ),
        )
