import elegy


import jax.numpy as jnp


#
def test_basic():

    y_true = jnp.array([1, 2])
    y_pred = jnp.array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])

    # Using 'auto'/'sum_over_batch_size' reduction type.
    scce = elegy.losses.SparseCategoricalCrossentropy()
    result = scce(y_true, y_pred)  # 1.177
    assert jnp.isclose(result, 1.177, rtol=0.01)

    # Calling with 'sample_weight'.
    result = scce(y_true, y_pred, sample_weight=jnp.array([0.3, 0.7]))  # 0.814
    assert jnp.isclose(result, 0.814, rtol=0.01)

    # Using 'sum' reduction type.
    scce = elegy.losses.SparseCategoricalCrossentropy(
        reduction=elegy.losses.Reduction.SUM
    )
    result = scce(y_true, y_pred)  # 2.354
    assert jnp.isclose(result, 2.354, rtol=0.01)

    # Using 'none' reduction type.
    scce = elegy.losses.SparseCategoricalCrossentropy(
        reduction=elegy.losses.Reduction.NONE
    )
    result = scce(y_true, y_pred)  # [0.0513, 2.303]
    assert jnp.all(jnp.isclose(result, [0.0513, 2.303], rtol=0.01))


def test_scce_out_of_bounds():
    ypred = jnp.zeros([4, 10])
    ytrue0 = jnp.array([0, 0, -1, 0])
    ytrue1 = jnp.array([0, 0, 10, 0])

    scce = elegy.losses.SparseCategoricalCrossentropy()

    assert jnp.isnan(scce(ytrue0, ypred)).any()
    assert jnp.isnan(scce(ytrue1, ypred)).any()

    scce = elegy.losses.SparseCategoricalCrossentropy(check_bounds=False)
    assert not jnp.isnan(scce(ytrue0, ypred)).any()
    assert not jnp.isnan(scce(ytrue1, ypred)).any()
