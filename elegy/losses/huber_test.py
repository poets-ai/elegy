import elegy


import jax.numpy as jnp
import jax
import tensorflow.keras as tfk

# import debugpy

# print("Waiting for debugger...")
# debugpy.listen(5679)
# debugpy.wait_for_client()


def test_basic():

    y_true = jnp.array([[0, 1], [0, 0]])
    y_pred = jnp.array([[0.6, 0.4], [0.4, 0.6]])

    # Using 'auto'/'sum_over_batch_size' reduction type.
    huber_loss = elegy.losses.Huber()
    assert huber_loss(y_true, y_pred) == 0.155

    # Calling with 'sample_weight'.
    assert huber_loss(y_true, y_pred, sample_weight=jnp.array([0.8, 0.2])) == 0.08500001

    # Using 'sum' reduction type.
    huber_loss = elegy.losses.Huber(reduction=elegy.losses.Reduction.SUM)
    assert huber_loss(y_true, y_pred) == 0.31

    # Using 'none' reduction type.
    huber_loss = elegy.losses.Huber(reduction=elegy.losses.Reduction.NONE)

    assert jnp.equal(huber_loss(y_true, y_pred), jnp.array([0.18, 0.13000001])).all()


def test_function():

    rng = jax.random.PRNGKey(42)

    y_true = jax.random.randint(rng, shape=(2, 3), minval=0, maxval=2)
    y_pred = jax.random.uniform(rng, shape=(2, 3))

    loss = elegy.losses.huber(y_true, y_pred, delta=1.0)
    assert loss.shape == (2,)

    y_pred = y_pred.astype(float)
    y_true = y_true.astype(float)
    delta = 1.0
    error = jnp.subtract(y_pred, y_true)
    abs_error = jnp.abs(error)
    quadratic = jnp.minimum(abs_error, delta)
    linear = jnp.subtract(abs_error, quadratic)
    assert jnp.array_equal(
        loss,
        jnp.mean(
            jnp.add(
                jnp.multiply(0.5, jnp.multiply(quadratic, quadratic)),
                jnp.multiply(delta, linear),
            ),
            axis=-1,
        ),
    )


def test_compatibility():
    # Input:  true (y_true) and predicted (y_pred) tensors
    rng = jax.random.PRNGKey(121)

    y_true = jax.random.randint(rng, shape=(2, 3), minval=0, maxval=2)
    y_true = y_true.astype(dtype=jnp.float32)
    y_pred = jax.random.uniform(rng, shape=(2, 3))

    # cosine_loss using sample_weight
    huber_loss = elegy.losses.Huber(delta=1.0)
    huber_loss_tfk = tfk.losses.Huber(delta=1.0)

    assert jnp.isclose(
        huber_loss(y_true, y_pred, sample_weight=jnp.array([1, 0])),
        huber_loss_tfk(y_true, y_pred, sample_weight=jnp.array([1, 0])),
        rtol=0.0001,
    )

    # cosine_loss with reduction method: SUM
    huber_loss = elegy.losses.Huber(delta=1.0, reduction=elegy.losses.Reduction.SUM)
    huber_loss_tfk = tfk.losses.Huber(delta=1.0, reduction=tfk.losses.Reduction.SUM)
    assert jnp.isclose(
        huber_loss(y_true, y_pred), huber_loss_tfk(y_true, y_pred), rtol=0.0001
    )

    # cosine_loss with reduction method: NONE
    huber_loss = elegy.losses.Huber(delta=1.0, reduction=elegy.losses.Reduction.NONE)
    huber_loss_tfk = tfk.losses.Huber(delta=1.0, reduction=tfk.losses.Reduction.NONE)
    assert jnp.all(
        jnp.isclose(
            huber_loss(y_true, y_pred), huber_loss_tfk(y_true, y_pred), rtol=0.0001
        )
    )


if __name__ == "__main__":

    test_basic()
    test_function()
    test_compatibility()
