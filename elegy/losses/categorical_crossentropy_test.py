import elegy


import jax.numpy as jnp


# import debugpy

# print("Waiting for debugger...")
# debugpy.listen(5677)
# debugpy.wait_for_client()


#
def test_basic():

    y_true = jnp.array([[0, 1, 0], [0, 0, 1]])
    y_pred = jnp.array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])

    # Using 'auto'/'sum_over_batch_size' reduction type.
    cce = elegy.losses.CategoricalCrossentropy()
    result = cce(y_true, y_pred)  # 1.77
    assert jnp.isclose(result, 1.177, rtol=0.01)

    # Calling with 'sample_weight'.
    result = cce(y_true, y_pred, sample_weight=jnp.array([0.3, 0.7]))  # 0.814
    assert jnp.isclose(result, 0.814, rtol=0.01)

    # Using 'sum' reduction type.
    cce = elegy.losses.CategoricalCrossentropy(reduction=elegy.losses.Reduction.SUM)
    result = cce(y_true, y_pred)  # 2.354
    assert jnp.isclose(result, 2.354, rtol=0.01)

    # Using 'none' reduction type.
    cce = elegy.losses.CategoricalCrossentropy(reduction=elegy.losses.Reduction.NONE)
    result = cce(y_true, y_pred)  # [0.0513, 2.303]
    assert jnp.all(jnp.isclose(result, [0.0513, 2.303], rtol=0.01))
