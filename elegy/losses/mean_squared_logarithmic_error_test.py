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

    y_true = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    y_pred = jnp.array([[1.0, 1.0], [1.0, 0.0]])

    # Using 'auto'/'sum_over_batch_size' reduction type.
    msle = elegy.losses.MeanSquaredLogarithmicError()

    assert msle(y_true, y_pred) == 0.24022643

    # Calling with 'sample_weight'.
    assert msle(y_true, y_pred, sample_weight=jnp.array([0.7, 0.3])) == 0.12011322

    # Using 'sum' reduction type.
    msle = elegy.losses.MeanSquaredLogarithmicError(
        reduction=elegy.losses.Reduction.SUM
    )

    assert msle(y_true, y_pred) == 0.48045287

    # Using 'none' reduction type.
    msle = elegy.losses.MeanSquaredLogarithmicError(
        reduction=elegy.losses.Reduction.NONE
    )

    assert jnp.equal(msle(y_true, y_pred), jnp.array([0.24022643, 0.24022643])).all()


def test_function():

    rng = jax.random.PRNGKey(42)

    y_true = jax.random.randint(rng, shape=(2, 3), minval=0, maxval=2)
    y_pred = jax.random.uniform(rng, shape=(2, 3))

    loss = elegy.losses.mean_squared_logarithmic_error(y_true, y_pred)

    assert loss.shape == (2,)

    first_log = jnp.log(jnp.maximum(y_true, types.EPSILON) + 1.0)
    second_log = jnp.log(jnp.maximum(y_pred, types.EPSILON) + 1.0)
    assert jnp.array_equal(loss, jnp.mean(jnp.square(first_log - second_log), axis=-1))


def test_compatibility():
    # Input:  true (y_true) and predicted (y_pred) tensors
    y_true = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    y_pred = jnp.array([[0.6, 0.4], [0.4, 0.6]])

    # MSLE using sample_weight
    msle_elegy = elegy.losses.MeanSquaredLogarithmicError()
    msle_tfk = tfk.losses.MeanSquaredLogarithmicError()
    assert jnp.isclose(
        msle_elegy(y_true, y_pred, sample_weight=jnp.array([1, 0])),
        msle_tfk(y_true, y_pred, sample_weight=jnp.array([1, 0])),
        rtol=0.0001,
    )

    # MSLE with reduction method: SUM
    msle_elegy = elegy.losses.MeanSquaredLogarithmicError(
        reduction=elegy.losses.Reduction.SUM
    )
    msle_tfk = tfk.losses.MeanSquaredLogarithmicError(
        reduction=tfk.losses.Reduction.SUM
    )
    assert jnp.isclose(
        msle_elegy(y_true, y_pred), msle_tfk(y_true, y_pred), rtol=0.0001
    )

    # MSLE with reduction method: NONE
    msle_elegy = elegy.losses.MeanSquaredLogarithmicError(
        reduction=elegy.losses.Reduction.NONE
    )
    msle_tfk = tfk.losses.MeanSquaredLogarithmicError(
        reduction=tfk.losses.Reduction.NONE
    )
    assert jnp.all(
        jnp.isclose(msle_elegy(y_true, y_pred), msle_tfk(y_true, y_pred), rtol=0.0001)
    )


if __name__ == "__main__":

    test_basic()
    test_function()
