import elegy
from haiku.testing import transform_and_run
import jax.numpy as jnp


@transform_and_run
def test_basic():
    y_true = jnp.array([[0., 1.], [0., 0.]])
    y_pred = jnp.array([[0.6, 0.4], [0.4, 0.6]])

    bce = elegy.losses.BinaryCrossEntropy()
    result = bce(y_true, y_pred)
    assert jnp.isclose(result, 0.815, rtol=0.01)

    result = bce(y_true, y_pred, sample_weight=jnp.array([1, 0]))
    assert jnp.isclose(result, 0.458, rtol=0.01)

    bce = elegy.losses.BinaryCrossEntropy(reduction=elegy.losses.Reduction.SUM)
    result = bce(y_true, y_pred)
    assert jnp.isclose(result, 1.630, rtol=0.01)

    bce = elegy.losses.BinaryCrossEntropy(reduction=elegy.losses.Reduction.NONE)
    result = bce(y_true, y_pred)
    assert jnp.all(jnp.isclose(result, [0.916, 0.713], rtol=0.01))

