from unittest import TestCase

import jax.numpy as jnp
import tensorflow.keras as tfk

import elegy


class BinaryCrossentropyTest(TestCase):
    #
    def test_basic(self):

        # Input:  true (y_true) and predicted (y_pred) tensors
        y_true = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        y_pred = jnp.array([[0.6, 0.4], [0.4, 0.6]])

        # Standard BCE, considering prediction tensor as probabilities
        bce = elegy.metrics.BinaryCrossentropy()
        result = bce.call_with_defaults()(
            y_true=y_true,
            y_pred=y_pred,
        )
        assert jnp.isclose(result, 0.815, rtol=0.01)

        # Standard BCE, considering prediction tensor as logits
        y_logits = jnp.log(y_pred) - jnp.log(1 - y_pred)
        bce = elegy.metrics.BinaryCrossentropy(from_logits=True)
        result_from_logits = bce.call_with_defaults()(y_true, y_logits)
        assert jnp.isclose(result_from_logits, 0.815, rtol=0.01)
        assert jnp.isclose(result_from_logits, result, rtol=0.01)

        # BCE using sample_weight
        bce = elegy.metrics.BinaryCrossentropy()
        result = bce.call_with_defaults()(
            y_true, y_pred, sample_weight=jnp.array([1.0, 0.0])
        )
        assert jnp.isclose(result, 0.916, rtol=0.01)

    #
    def test_compatibility(self):

        # Input:  true (y_true) and predicted (y_pred) tensors
        y_true = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        y_pred = jnp.array([[0.6, 0.4], [0.4, 0.6]])

        # Standard BCE, considering prediction tensor as probabilities
        bce_elegy = elegy.metrics.BinaryCrossentropy()
        bce_tfk = tfk.metrics.BinaryCrossentropy()
        assert jnp.isclose(
            bce_elegy.call_with_defaults()(y_true, y_pred),
            bce_tfk(y_true, y_pred),
            rtol=0.0001,
        )

        # Standard BCE, considering prediction tensor as logits
        y_logits = jnp.log(y_pred) - jnp.log(1 - y_pred)
        bce_elegy = elegy.metrics.BinaryCrossentropy(from_logits=True)
        bce_tfk = tfk.metrics.BinaryCrossentropy(from_logits=True)
        assert jnp.isclose(
            bce_elegy.call_with_defaults()(y_true, y_logits),
            bce_tfk(y_true, y_logits),
            rtol=0.0001,
        )

        # BCE using sample_weight
        bce_elegy = elegy.metrics.BinaryCrossentropy()
        bce_tfk = tfk.metrics.BinaryCrossentropy()
        assert jnp.isclose(
            bce_elegy.call_with_defaults()(
                y_true, y_pred, sample_weight=jnp.array([1, 0])
            ),
            bce_tfk(y_true, y_pred, sample_weight=jnp.array([1, 0])),
            rtol=0.0001,
        )
