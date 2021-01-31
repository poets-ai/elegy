import elegy
import jax.numpy as jnp


def test_basic():

    mse = elegy.metrics.MeanSquaredError()

    result = mse.call_with_defaults()(
        y_true=jnp.array([1, 1, 1, 1]), y_pred=jnp.array([0, 1, 1, 1])
    )
    assert result == 0.25

    result = mse.call_with_defaults()(
        y_true=jnp.array([1, 1, 1, 1]), y_pred=jnp.array([1, 0, 0, 0])
    )
    assert result == 0.5
