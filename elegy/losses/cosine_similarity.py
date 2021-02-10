from elegy import types
import typing as tp

import jax.numpy as jnp

from elegy import utils, types
from elegy.losses.loss import Loss, Reduction


def cosine_similarity(
    y_true: jnp.ndarray, y_pred: jnp.ndarray, axis: int
) -> jnp.ndarray:
    """
    Computes the cosine similarity between labels and predictions.

    ```python
    loss = -sum(l2_norm(y_true) * l2_norm(y_pred))
    ```

    Usage:

    ```python
    rng = jax.random.PRNGKey(42)

    y_true = jax.random.randint(rng, shape=(2, 3), minval=0, maxval=2)
    y_pred = jax.random.uniform(rng, shape=(2, 3))

    loss = elegy.losses.cosine_similarity(y_true, y_pred, axis=1)
    assert loss.shape == (2,)

    y_true = y_true / jnp.maximum(jnp.linalg.norm(y_true, axis=1, keepdims=True), jnp.sqrt(types.EPSILON))
    y_pred = y_pred / jnp.maximum(jnp.linalg.norm(y_pred, axis=1, keepdims=True), jnp.sqrt(types.EPSILON))
    assert jnp.array_equal(loss, -jnp.sum(y_true * y_pred, axis=1))
    ```

    Arguments:
        y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
        axis: The dimension along which the cosine similarity is computed.

    Returns:
          cosine similarity Values. If reduction is NONE, this has
         shape [batch_size, d0, .. dN-1]; otherwise, it is scalar.
         (Note dN-1 because all loss functions reduce by 1 dimension, usually axis=-1.)
    """
    y_true = y_true / jnp.maximum(
        jnp.linalg.norm(y_true, axis=axis, keepdims=True), jnp.sqrt(types.EPSILON)
    )
    y_pred = y_pred / jnp.maximum(
        jnp.linalg.norm(y_pred, axis=axis, keepdims=True), jnp.sqrt(types.EPSILON)
    )
    return -jnp.sum(y_true * y_pred, axis=axis)


class CosineSimilarity(Loss):
    """
    Computes the mean squared logarithmic errors between labels and predictions.

    `loss = -sum(l2_norm(y_true) * l2_norm(y_pred))`

    Usage:

    ```python
    y_true = jnp.array([[0., 1.], [1., 1.]])
    y_pred = jnp.array([[1., 0.], [1., 1.]])

    # Using 'auto'/'sum_over_batch_size' reduction type.
    cosine_loss = elegy.losses.CosineSimilarity(axis=1)
    assert cosine_loss(y_true, y_pred) == -0.49999997

    # Calling with 'sample_weight'.
    assert cosine_loss(y_true, y_pred, sample_weight=jnp.array([0.8, 0.2])) == -0.099999994

    # Using 'sum' reduction type.
    cosine_loss = elegy.losses.CosineSimilarity(axis=1,
        reduction=elegy.losses.Reduction.SUM
    )
    assert cosine_loss(y_true, y_pred) == -0.99999994

    # Using 'none' reduction type.
    cosine_loss = elegy.losses.CosineSimilarity(axis=1,
        reduction=elegy.losses.Reduction.NONE
    )

    assert jnp.equal(cosine_loss(y_true, y_pred), jnp.array([-0., -0.99999994])).all()
    ```
    Usage with the Elegy API:

    ```python
    model = elegy.Model(
        module_fn,
        loss=elegy.losses.CosineSimilarity(axis=1),
        metrics=elegy.metrics.Mean(),
    )
    ```
    """

    def __init__(
        self,
        axis: int = -1,
        reduction: tp.Optional[Reduction] = None,
        weight: tp.Optional[float] = None,
        on: tp.Optional[types.IndexLike] = None,
        **kwargs
    ):
        """
        Initializes `Mean` class.

        Arguments:
            axis: (Optional) Defaults to -1. The dimension along which the cosine
                   similarity is computed.
            reduction: (Optional) Type of `elegy.losses.Reduction` to apply to
                loss. Default value is `SUM_OVER_BATCH_SIZE`. For almost all cases
                this defaults to `SUM_OVER_BATCH_SIZE`.
            weight: Optional weight contribution for the total loss. Defaults to `1`.
            on: A string or integer, or iterable of string or integers, that
                indicate how to index/filter the `y_true` and `y_pred`
                arguments before passing them to `call`. For example if `on = "a"` then
                `y_true = y_true["a"]`. If `on` is an iterable
                the structures will be indexed iteratively, for example if `on = ["a", 0, "b"]`
                then `y_true = y_true["a"][0]["b"]`, same for `y_pred`. For more information
                check out [Keras-like behavior](https://poets-ai.github.io/elegy/guides/modules-losses-metrics/#keras-like-behavior).
        """
        self.axis = axis
        return super().__init__(reduction=reduction, weight=weight, on=on, **kwargs)

    def call(
        self,
        y_true: jnp.ndarray,
        y_pred: jnp.ndarray,
        sample_weight: tp.Optional[
            jnp.ndarray
        ] = None,  # not used, __call__ handles it, left for documentation purposes.
    ) -> jnp.ndarray:
        """
        Invokes the `CosineSimilarity` instance.

        Arguments:
            y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`, except
                sparse loss functions such as sparse categorical crossentropy where
                shape = `[batch_size, d0, .. dN-1]`
            y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`
            sample_weight: Optional `sample_weight` acts as a
                coefficient for the loss. If a scalar is provided, then the loss is
                simply scaled by the given value. If `sample_weight` is a tensor of size
                `[batch_size]`, then the total loss for each sample of the batch is
                rescaled by the corresponding element in the `sample_weight` vector. If
                the shape of `sample_weight` is `[batch_size, d0, .. dN-1]` (or can be
                broadcasted to this shape), then each loss element of `y_pred` is scaled
                by the corresponding value of `sample_weight`. (Note on`dN-1`: all loss
                functions reduce by 1 dimension, usually axis=-1.)

        Returns:
            Weighted loss float `Tensor`. If `reduction` is `NONE`, this has
                shape `[batch_size, d0, .. dN-1]`; otherwise, it is scalar. (Note `dN-1`
                because all loss functions reduce by 1 dimension, usually axis=-1.)

        Raises:
            ValueError: If the shape of `sample_weight` is invalid.
        """
        return cosine_similarity(y_true, y_pred, self.axis)
