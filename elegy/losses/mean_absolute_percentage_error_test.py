import elegy
import haiku as hk
from haiku.testing import transform_and_run
import jax.numpy as jnp
import jax
import utils


# import debugpy

# print("Waiting for debugger...")
# debugpy.listen(5679)
# debugpy.wait_for_client()


@transform_and_run
def test_basic():

    y_true = jnp.array([[1.0, 1.0], [0.9, 0.0]])
    y_pred = jnp.array([[1.0, 1.0], [1.0, 0.0]])

    # Using 'auto'/'sum_over_batch_size' reduction type.
    mape = elegy.losses.MeanAbsolutePercentageError()
    
    assert mape(y_true, y_pred) == 2.7777786

    # Calling with 'sample_weight'.
    assert mape(y_true, y_pred, sample_weight=jnp.array([0.1, 0.9])) == 2.5000007

    # Using 'sum' reduction type.
    mape = elegy.losses.MeanAbsolutePercentageError(reduction=elegy.losses.Reduction.SUM)

    assert mape(y_true, y_pred) == 5.5555573

    # Using 'none' reduction type.
    mape = elegy.losses.MeanAbsolutePercentageError(reduction=elegy.losses.Reduction.NONE)

    assert list(mape(y_true, y_pred)) == [0. , 5.5555573]


def test_function():

    rng = jax.random.PRNGKey(42)

    y_true = jax.random.randint(rng, shape=(2, 3), minval=0, maxval=2)
    y_pred = jax.random.uniform(rng, shape=(2, 3))

    loss = elegy.losses.mean_absolute_error(y_true, y_pred)

    assert loss.shape == (2,)

    assert jnp.array_equal(loss, 100 * jnp.mean(jnp.abs((y_pred - y_true) / (jnp.clip(y_true, utils.EPSILON, None))), axis=-1))


if __name__ == '__main__':

    test_basic()
    test_function()