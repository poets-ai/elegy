from elegy import types
import typing as tp

import jax.numpy as jnp

from elegy.losses.loss import Loss, Reduction


def huber(y_true: jnp.ndarray, y_pred: jnp.ndarray, delta: float) -> jnp.ndarray:
    r"""
    Computes the Huber loss between labels and predictions.
    
    For each value x in error = y_true - y_pred:

    $$
    loss =
    \begin{cases}
    \ 0.5 \times x^2,\hskip8em\text{if } |x|\leq d\\
    0.5 \times d^2 + d \times (|x| - d),\hskip1.7em \text{otherwise} 
    \end{cases}
    $$
    
    where d is delta. See: https://en.wikipedia.org/wiki/Huber_loss

    Usage:

    ```python
    rng = jax.random.PRNGKey(42)

    y_true = jax.random.randint(rng, shape=(2, 3), minval=0, maxval=2)
    y_pred = jax.random.uniform(rng, shape=(2, 3))

    loss = elegy.losses.huber(y_true, y_pred, delta=1.0)
    assert loss.shape == (2,)

    y_pred = y_pred.astype(float)
    y_true = y_true.astype(float)
    delta = 1.0
    error = jnp.subtract(y_pred, y_true)
    abs_error = jnp.abs(error)
    quadratic = jnp.minimum(abs_error, delta)
    linear = jnp.subtract(abs_error, quadratic)
    assert jnp.array_equal(loss, jnp.mean(
      jnp.add(
          jnp.multiply(
              0.5,
              jnp.multiply(quadratic, quadratic)
              ),
              jnp.multiply(delta, linear)), axis=-1
    ))
    ```

    Arguments:
        y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
        delta: A float, the point where the Huber loss function changes from a quadratic to linear.

    Returns:
          huber loss Values. If reduction is NONE, this has
         shape [batch_size, d0, .. dN-1]; otherwise, it is scalar.
         (Note dN-1 because all loss functions reduce by 1 dimension, usually axis=-1.)
    """
    y_pred = y_pred.astype(float)
    y_true = y_true.astype(float)
    delta = float(delta)
    error = jnp.subtract(y_pred, y_true)
    abs_error = jnp.abs(error)
    quadratic = jnp.minimum(abs_error, delta)
    linear = jnp.subtract(abs_error, quadratic)
    return jnp.mean(
        jnp.add(
            jnp.multiply(0.5, jnp.multiply(quadratic, quadratic)),
            jnp.multiply(delta, linear),
        ),
        axis=-1,
    )


class Huber(Loss):
    r"""
    Computes the Huber loss  between labels and predictions.
    
    For each value x in error = y_true - y_pred:

    $$
    loss =
    \begin{cases}
    \ 0.5 \times x^2,\hskip8em\text{if } |x|\leq d\\
    0.5 \times d^2 + d \times (|x| - d),\hskip1.7em \text{otherwise} 
    \end{cases}
    $$
    
    where d is delta. See: https://en.wikipedia.org/wiki/Huber_loss

    Usage:

    ```python
    y_true = jnp.array([[0, 1], [0, 0]])
    y_pred = jnp.array([[0.6, 0.4], [0.4, 0.6]])

    # Using 'auto'/'sum_over_batch_size' reduction type.
    huber_loss = elegy.losses.Huber()
    assert huber_loss(y_true, y_pred) == 0.155

    # Calling with 'sample_weight'.
    assert (
        huber_loss(y_true, y_pred, sample_weight=jnp.array([0.8, 0.2])) == 0.08500001
    )

    # Using 'sum' reduction type.
    huber_loss = elegy.losses.Huber(
        reduction=elegy.losses.Reduction.SUM
    )
    assert huber_loss(y_true, y_pred) == 0.31

    # Using 'none' reduction type.
    huber_loss = elegy.losses.Huber(
        reduction=elegy.losses.Reduction.NONE
    )

    assert jnp.equal(huber_loss(y_true, y_pred), jnp.array([0.18, 0.13000001])).all()
    ```
    Usage with the Elegy API:

    ```python
    model = elegy.Model(
        module_fn,
        loss=elegy.losses.Huber(delta=1.0),
        metrics=elegy.metrics.Mean(),
    )
    ```
    """

    def __init__(
        self,
        delta: float = 1.0,
        reduction: tp.Optional[Reduction] = None,
        weight: tp.Optional[float] = None,
        on: tp.Optional[types.IndexLike] = None,
        **kwargs
    ):
        """
        Initializes `Mean` class.

        Arguments:
            delta: (Optional) Defaults to 1.0. A float, the point where the Huber loss function changes from a quadratic to linear.
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
        self.delta = delta
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
        Invokes the `Huber` instance.

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
        return huber(y_true, y_pred, self.delta)
