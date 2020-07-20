import elegy
from haiku.testing import transform_and_run
import jax.numpy as jnp
import numpy as np
from elegy import utils


@transform_and_run
def test_basic():
    y_true = jnp.array([[0., 1.], [0., 0.]])
    y_pred = jnp.array([[0.6, 0.4], [0.4, 0.6]])
    bce = elegy.losses.BinaryCrossEntropy()
    result = bce(y_true, y_pred)
    assert jnp.isclose(result, 0.458, rtol=0.01)

