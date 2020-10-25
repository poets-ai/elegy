import elegy

from elegy.testing_utils import transform_and_run
from elegy import utils
import jax.numpy as jnp
import jax
from jax.lax import rsqrt
import tensorflow.keras as tfk

# import debugpy

# print("Waiting for debugger...")
# debugpy.listen(5679)
# debugpy.wait_for_client()


@transform_and_run
def test_basic():

    y_true = jnp.array([[0., 1.], [1., 1.]])
    y_pred = jnp.array([[1., 0.], [1., 1.]])

    # Using 'auto'/'sum_over_batch_size' reduction type.
    cosine_loss = elegy.losses.CosineSimilarity(axis=1)
    assert cosine_loss(y_true, y_pred) == -0.49999997

    # Calling with 'sample_weight'.
    assert cosine_loss(y_true, y_pred, sample_weight=jnp.array([0.8, 0.2])) == -0.099999994

    # Using 'sum' reduction type.
    cosine_loss = elegy.losses.CosineSimilarity(axis=1,
        reduction=elegy.losses.Reduction.SUM
    )
    assert cosine_loss(y_true, y_pred) == -0.99999994

    # Using 'none' reduction type.
    cosine_loss = elegy.losses.CosineSimilarity(axis=1,
        reduction=elegy.losses.Reduction.NONE
    )

    assert jnp.equal(cosine_loss(y_true, y_pred), jnp.array([-0., -0.99999994])).all()


@transform_and_run
def test_function():

    rng = jax.random.PRNGKey(42)

    y_true = jax.random.randint(rng, shape=(2, 3), minval=0, maxval=2)
    y_pred = jax.random.uniform(rng, shape=(2, 3))
    
    def _l2_normalize(x, axis=None, epsilon=utils.EPSILON):
        square_sum = jnp.sum(jnp.square(x), axis=axis, keepdims=True)
        x_inv_norm = rsqrt(jnp.maximum(square_sum, epsilon))
        return jnp.multiply(x, x_inv_norm)

    loss = elegy.losses.cosine_similarity(y_true, y_pred, axis=1)

    assert loss.shape == (2,)

    y_true = _l2_normalize(y_true, axis=1)
    y_pred = _l2_normalize(y_pred, axis=1)
    assert jnp.array_equal(loss, -jnp.sum(y_true * y_pred, axis=1))


@transform_and_run
def test_compatibility():
    # Input:  true (y_true) and predicted (y_pred) tensors
    y_true = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    y_pred = jnp.array([[0.6, 0.4], [0.4, 0.6]])

    # cosine_loss using sample_weight
    cosine_loss = elegy.losses.CosineSimilarity(axis=1)
    cosine_loss_tfk = tfk.losses.CosineSimilarity(axis=1)
    assert jnp.isclose(
        cosine_loss(y_true, y_pred, sample_weight=jnp.array([1, 0])),
        cosine_loss_tfk(y_true, y_pred, sample_weight=jnp.array([1, 0])),
        rtol=0.0001,
    )

    # cosine_loss with reduction method: SUM
    cosine_loss = elegy.losses.CosineSimilarity(axis=1,
        reduction=elegy.losses.Reduction.SUM
    )
    cosine_loss_tfk = tfk.losses.CosineSimilarity(axis=1,
        reduction=tfk.losses.Reduction.SUM
    )
    assert jnp.isclose(
        cosine_loss(y_true, y_pred), cosine_loss_tfk(y_true, y_pred), rtol=0.0001
    )

    # cosine_loss with reduction method: NONE
    cosine_loss = elegy.losses.CosineSimilarity(axis=1,
        reduction=elegy.losses.Reduction.NONE
    )
    cosine_loss_tfk = tfk.losses.CosineSimilarity(axis=1,
        reduction=tfk.losses.Reduction.NONE
    )
    assert jnp.all(
        jnp.isclose(cosine_loss(y_true, y_pred), cosine_loss_tfk(y_true, y_pred), rtol=0.0001)
    )


if __name__ == "__main__":

    test_basic()
    test_function()
    test_compatibility()