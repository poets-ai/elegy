import elegy


import jax.numpy as jnp
import jax


# import debugpy

# print("Waiting for debugger...")
# debugpy.listen(5679)
# debugpy.wait_for_client()


#
def test_basic():

    y_true = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    y_pred = jnp.array([[1.0, 1.0], [1.0, 0.0]])

    # Using 'auto'/'sum_over_batch_size' reduction type.
    mae = elegy.losses.MeanAbsoluteError()

    assert mae(y_true, y_pred) == 0.5

    # Calling with 'sample_weight'.
    assert mae(y_true, y_pred, sample_weight=jnp.array([0.7, 0.3])) == 0.25

    # Using 'sum' reduction type.
    mae = elegy.losses.MeanAbsoluteError(reduction=elegy.losses.Reduction.SUM)

    assert mae(y_true, y_pred) == 1.0

    # Using 'none' reduction type.
    mae = elegy.losses.MeanAbsoluteError(reduction=elegy.losses.Reduction.NONE)

    assert list(mae(y_true, y_pred)) == [0.5, 0.5]


#
def test_function():

    rng = jax.random.PRNGKey(42)

    y_true = jax.random.randint(rng, shape=(2, 3), minval=0, maxval=2)
    y_pred = jax.random.uniform(rng, shape=(2, 3))

    loss = elegy.losses.mean_absolute_error(y_true, y_pred)

    assert loss.shape == (2,)

    assert jnp.array_equal(loss, jnp.mean(jnp.abs(y_true - y_pred), axis=-1))


if __name__ == "__main__":

    test_basic()
    test_function()
