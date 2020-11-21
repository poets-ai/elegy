import elegy


import jax.numpy as jnp


# import debugpy

# print("Waiting for debugger...")
# debugpy.listen(5679)
# debugpy.wait_for_client()


#
def test_basic():

    mse = elegy.metrics.MeanSquaredError()

    result = mse(y_true=jnp.array([1, 1, 1, 1]), y_pred=jnp.array([0, 1, 1, 1]))
    assert result == 0.25

    result = mse(y_true=jnp.array([1, 1, 1, 1]), y_pred=jnp.array([1, 0, 0, 0]))
    assert result == 0.5
