import elegy


from elegy import utils, types
import jax.numpy as jnp
import jax
import tensorflow.keras as tfk

# import debugpy

# print("Waiting for debugger...")
# debugpy.listen(5679)
# debugpy.wait_for_client()


def test_basic():

    y_true = jnp.array([[0.0, 1.0], [1.0, 1.0]])
    y_pred = jnp.array([[1.0, 0.0], [1.0, 1.0]])

    # Using 'auto'/'sum_over_batch_size' reduction type.
    cosine_loss = elegy.losses.CosineSimilarity(axis=1)
    assert cosine_loss(y_true, y_pred) == -0.49999997

    # Calling with 'sample_weight'.
    assert (
        cosine_loss(y_true, y_pred, sample_weight=jnp.array([0.8, 0.2])) == -0.099999994
    )

    # Using 'sum' reduction type.
    cosine_loss = elegy.losses.CosineSimilarity(
        axis=1, reduction=elegy.losses.Reduction.SUM
    )
    assert cosine_loss(y_true, y_pred) == -0.99999994

    # Using 'none' reduction type.
    cosine_loss = elegy.losses.CosineSimilarity(
        axis=1, reduction=elegy.losses.Reduction.NONE
    )

    assert jnp.equal(cosine_loss(y_true, y_pred), jnp.array([-0.0, -0.99999994])).all()


def test_function():

    rng = jax.random.PRNGKey(42)

    y_true = jax.random.randint(rng, shape=(2, 3), minval=0, maxval=2)
    y_pred = jax.random.uniform(rng, shape=(2, 3))

    loss = elegy.losses.cosine_similarity(y_true, y_pred, axis=1)
    assert loss.shape == (2,)

    y_true = y_true / jnp.maximum(
        jnp.linalg.norm(y_true, axis=1, keepdims=True), jnp.sqrt(types.EPSILON)
    )
    y_pred = y_pred / jnp.maximum(
        jnp.linalg.norm(y_pred, axis=1, keepdims=True), jnp.sqrt(types.EPSILON)
    )
    assert jnp.array_equal(loss, -jnp.sum(y_true * y_pred, axis=1))


def test_compatibility():
    # Input:  true (y_true) and predicted (y_pred) tensors
    rng = jax.random.PRNGKey(121)

    y_true = jax.random.randint(rng, shape=(2, 3), minval=0, maxval=2)
    y_true = y_true.astype(dtype=jnp.float32)
    y_pred = jax.random.uniform(rng, shape=(2, 3))

    # cosine_loss using sample_weight
    cosine_loss = elegy.losses.CosineSimilarity(axis=1)
    cosine_loss_tfk = tfk.losses.CosineSimilarity(axis=1)

    assert jnp.isclose(
        cosine_loss(y_true, y_pred, sample_weight=jnp.array([1, 0])),
        cosine_loss_tfk(y_true, y_pred, sample_weight=jnp.array([1, 0])),
        rtol=0.0001,
    )

    # cosine_loss with reduction method: SUM
    cosine_loss = elegy.losses.CosineSimilarity(
        axis=1, reduction=elegy.losses.Reduction.SUM
    )
    cosine_loss_tfk = tfk.losses.CosineSimilarity(
        axis=1, reduction=tfk.losses.Reduction.SUM
    )
    assert jnp.isclose(
        cosine_loss(y_true, y_pred), cosine_loss_tfk(y_true, y_pred), rtol=0.0001
    )

    # cosine_loss with reduction method: NONE
    cosine_loss = elegy.losses.CosineSimilarity(
        axis=1, reduction=elegy.losses.Reduction.NONE
    )
    cosine_loss_tfk = tfk.losses.CosineSimilarity(
        axis=1, reduction=tfk.losses.Reduction.NONE
    )
    assert jnp.all(
        jnp.isclose(
            cosine_loss(y_true, y_pred), cosine_loss_tfk(y_true, y_pred), rtol=0.0001
        )
    )


if __name__ == "__main__":

    test_basic()
    test_function()
    test_compatibility()
