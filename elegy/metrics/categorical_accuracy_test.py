import elegy


import jax.numpy as jnp


def test_basic():

    accuracy = elegy.metrics.CategoricalAccuracy()

    result = accuracy.call_with_defaults()(
        y_true=jnp.array([[0, 0, 1], [0, 1, 0]]),
        y_pred=jnp.array([[0.1, 0.9, 0.8], [0.05, 0.95, 0]]),
    )
    assert result == 0.5  # 1/2

    result = accuracy.call_with_defaults()(
        y_true=jnp.array([[0, 1, 0], [0, 1, 0]]),
        y_pred=jnp.array([[0.1, 0.9, 0.8], [0.05, 0.95, 0]]),
    )
    assert result == 0.75  # 3/4
