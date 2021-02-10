import elegy
import jax.numpy as jnp
import tensorflow.keras as tfk


def test_basic():
    y_true = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    y_pred = jnp.array([[0.6, 0.4], [0.4, 0.6]])

    bce = elegy.losses.BinaryCrossentropy()
    result = bce(y_true, y_pred)
    assert jnp.isclose(result, 0.815, rtol=0.01)

    y_logits = jnp.log(y_pred) - jnp.log(1 - y_pred)
    bce = elegy.losses.BinaryCrossentropy(from_logits=True)
    result_from_logits = bce(y_true, y_logits)
    assert jnp.isclose(result_from_logits, 0.815, rtol=0.01)
    assert jnp.isclose(result_from_logits, result, rtol=0.01)

    bce = elegy.losses.BinaryCrossentropy()
    result = bce(y_true, y_pred, sample_weight=jnp.array([1, 0]))
    assert jnp.isclose(result, 0.458, rtol=0.01)

    bce = elegy.losses.BinaryCrossentropy(reduction=elegy.losses.Reduction.SUM)
    result = bce(y_true, y_pred)
    assert jnp.isclose(result, 1.630, rtol=0.01)

    bce = elegy.losses.BinaryCrossentropy(reduction=elegy.losses.Reduction.NONE)
    result = bce(y_true, y_pred)
    assert jnp.all(jnp.isclose(result, [0.916, 0.713], rtol=0.01))


#
def test_compatibility():

    # Input:  true (y_true) and predicted (y_pred) tensors
    y_true = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    y_pred = jnp.array([[0.6, 0.4], [0.4, 0.6]])

    # Standard BCE, considering prediction tensor as probabilities
    bce_elegy = elegy.losses.BinaryCrossentropy()
    bce_tfk = tfk.losses.BinaryCrossentropy()
    assert jnp.isclose(bce_elegy(y_true, y_pred), bce_tfk(y_true, y_pred), rtol=0.0001)

    # Standard BCE, considering prediction tensor as logits
    y_logits = jnp.log(y_pred) - jnp.log(1 - y_pred)
    bce_elegy = elegy.losses.BinaryCrossentropy(from_logits=True)
    bce_tfk = tfk.losses.BinaryCrossentropy(from_logits=True)
    assert jnp.isclose(
        bce_elegy(y_true, y_logits), bce_tfk(y_true, y_logits), rtol=0.0001
    )

    # BCE using sample_weight
    bce_elegy = elegy.losses.BinaryCrossentropy()
    bce_tfk = tfk.losses.BinaryCrossentropy()
    assert jnp.isclose(
        bce_elegy(y_true, y_pred, sample_weight=jnp.array([1, 0])),
        bce_tfk(y_true, y_pred, sample_weight=jnp.array([1, 0])),
        rtol=0.0001,
    )

    # BCE with reduction method: SUM
    bce_elegy = elegy.losses.BinaryCrossentropy(reduction=elegy.losses.Reduction.SUM)
    bce_tfk = tfk.losses.BinaryCrossentropy(reduction=tfk.losses.Reduction.SUM)
    assert jnp.isclose(bce_elegy(y_true, y_pred), bce_tfk(y_true, y_pred), rtol=0.0001)

    # BCE with reduction method: NONE
    bce_elegy = elegy.losses.BinaryCrossentropy(reduction=elegy.losses.Reduction.NONE)
    bce_tfk = tfk.losses.BinaryCrossentropy(reduction=tfk.losses.Reduction.NONE)
    assert jnp.all(
        jnp.isclose(bce_elegy(y_true, y_pred), bce_tfk(y_true, y_pred), rtol=0.0001)
    )

    # BCE with label smoothing
    bce_elegy = elegy.losses.BinaryCrossentropy(label_smoothing=0.9)
    bce_tfk = tfk.losses.BinaryCrossentropy(label_smoothing=0.9)
    assert jnp.isclose(bce_elegy(y_true, y_pred), bce_tfk(y_true, y_pred), rtol=0.0001)
