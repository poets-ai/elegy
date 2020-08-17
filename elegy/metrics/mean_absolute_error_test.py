import elegy

from elegy.testing_utils import transform_and_run
import jax.numpy as jnp


# import debugpy

# print("Waiting for debugger...")
# debugpy.listen(5679)
# debugpy.wait_for_client()


@transform_and_run
def test_basic():

    mae = elegy.metrics.MeanAbsoluteError()

    result = mae(y_true=jnp.array([1, 1, 1, 1]), y_pred=jnp.array([0, 1, 1, 1]))
    assert result == 0.25

    result = mae(y_true=jnp.array([1, 1, 1, 1]), y_pred=jnp.array([1, 0, 0, 0]))
    assert result == 0.5
