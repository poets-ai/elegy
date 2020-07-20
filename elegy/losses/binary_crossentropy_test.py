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
    assert jnp.isclose(result, 1.630, rtol=0.01)


if __name__ == '__main__':
    ground_truth = jnp.array([[0., 1.], [0., 0.]])
    scores = jnp.array([[0.6, 0.4], [0.4, 0.6]])
    score = jnp.sum(ground_truth*jnp.log(scores)
                    + (1-ground_truth)*jnp.log(1-scores))
    result = (-1 / len(ground_truth)) * score
    print(result)



