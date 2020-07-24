import elegy
import haiku as hk
from haiku.testing import transform_and_run
import jax.numpy as jnp
import tensorflow.keras as tfk


@transform_and_run
def test_basic():

    # Input:  true (y_true) and predicted (y_pred) tensors
    y_true = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    y_pred = jnp.array([[0.6, 0.4], [0.4, 0.6]])

    # Standard BCE, considering prediction tensor as probabilities
    bce = elegy.metrics.BinaryCrossentropy()
    result = bce(y_true=y_true, y_pred=y_pred,)
    assert jnp.isclose(result, 0.815, rtol=0.01)

    # Standard BCE, considering prediction tensor as logits
    y_logits = jnp.log(y_pred) - jnp.log(1 - y_pred)
    bce = elegy.metrics.BinaryCrossentropy(from_logits=True)
    result_from_logits = bce(y_true, y_logits)
    assert jnp.isclose(result_from_logits, 0.815, rtol=0.01)
    assert jnp.isclose(result_from_logits, result, rtol=0.01)

    # BCE using sample_weight
    bce = elegy.metrics.BinaryCrossentropy()
    result = bce(y_true, y_pred, sample_weight=jnp.array([1.0, 0.0]))
    assert jnp.isclose(result, 0.916, rtol=0.01)


@transform_and_run
def test_compatibility():

    # Input:  true (y_true) and predicted (y_pred) tensors
    y_true = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    y_pred = jnp.array([[0.6, 0.4], [0.4, 0.6]])

    # Standard BCE, considering prediction tensor as probabilities
    bce_elegy = elegy.metrics.BinaryCrossentropy()
    bce_tfk = tfk.metrics.BinaryCrossentropy()
    assert jnp.isclose(bce_elegy(y_true, y_pred), bce_tfk(y_true, y_pred), rtol=0.0001)

    # Standard BCE, considering prediction tensor as logits
    y_logits = jnp.log(y_pred) - jnp.log(1 - y_pred)
    bce_elegy = elegy.metrics.BinaryCrossentropy(from_logits=True)
    bce_tfk = tfk.metrics.BinaryCrossentropy(from_logits=True)
    assert jnp.isclose(
        bce_elegy(y_true, y_logits), bce_tfk(y_true, y_logits), rtol=0.0001
    )

    # BCE using sample_weight
    bce_elegy = elegy.metrics.BinaryCrossentropy()
    bce_tfk = tfk.metrics.BinaryCrossentropy()
    assert jnp.isclose(
        bce_elegy(y_true, y_pred, sample_weight=jnp.array([1, 0])),
        bce_tfk(y_true, y_pred, sample_weight=jnp.array([1, 0])),
        rtol=0.0001,
    )


if __name__ == "__main__":
    test_basic()
    test_compatibility()
