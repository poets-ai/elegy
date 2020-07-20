from elegy import types
import typing as tp
import jax
import jax.numpy as jnp
from elegy import utils
from elegy.losses.loss import Loss, Reduction


def binary_cross_entropy(
    y_true: jnp.ndarray, y_pred: jnp.ndarray, from_logits: bool = False
) -> jnp.ndarray:
    if from_logits:
        y_pred = jax.nn.log_sigmoid(y_pred)
    '''
    Note, we add a very small value (in this case 1E-15) to the predicted probabilities to 
    avoid ever calculating the log of 0.0. This means that in practice, 
    the best possible loss will be a value very close to zero, but not exactly zero.
    '''
    # implementation 1
    #score = jnp.sum(y_true * jnp.log(utils.EPSILON + y_pred))
    #mean_sum_score = 1.0 / len(y_true) * score
    #return -jnp.array(mean_sum_score)

    # implementation 2
    score = jnp.sum(y_true * jnp.log(y_pred)
                    + (1 - y_true) * jnp.log(1 - y_pred))
    return jnp.array((-1 / len(y_true)) * score)



class BinaryCrossEntropy(Loss):
    """Computes the cross-entropy loss between true labels and predicted labels.
      Use this cross-entropy loss when there are only two label classes (assumed to
      be 0 and 1). For each example, there should be a single floating-point value
      per prediction.
      In the snippet below, each of the four examples has only a single
      floating-pointing value, and both `y_pred` and `y_true` have the shape
      `[batch_size]`.
      Standalone usage:
      >>> y_true = [[0., 1.], [0., 0.]]
      >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
      >>> # Using 'auto'/'sum_over_batch_size' reduction type.
      >>> bce = elegy.losses.BinaryCrossEntropy()
      >>> bce(y_true, y_pred).numpy()
      0.815
      >>> # Calling with 'sample_weight'.
      >>> bce(y_true, y_pred, sample_weight=[1, 0]).numpy()
      0.458
       >>> # Using 'sum' reduction type.
      >>> bce = elegy.losses.BinaryCrossEntropy()(
      ...     reduction=elegy.losses.Reduction.SUM
      >>> bce(y_true, y_pred).numpy()
      1.630
      >>> # Using 'none' reduction type.
      >>> bce = elegy.losses.BinaryCrossEntropy(
      ...     reduction=elegy.losses.Reduction.NONE)
      >>> bce(y_true, y_pred).numpy()
      array([0.916 , 0.714], dtype=float32)
       Usage with the `compile` API:
      ```python
       model = elegy.Model(
        module_fn,
        loss=elegy.losses.BinaryCrossEntropy(),
        metrics=elegy.metrics.Accuracy.defer(),
        optimizer=optix.adam(1e-3),
    )
      ```
      """
    def __init__(
            self,
            from_logits=False,
            label_smoothing: float=0,
            reduction: tp.Optional[Reduction] = None,
            name: tp.Optional[str] = None
    ):
        super().__init__(reduction=reduction, name=name)
        self._from_logits = from_logits
        self._label_smoothing = label_smoothing

    def call(
            self,
            y_true: jnp.ndarray,
            y_pred: jnp.ndarray,
            sample_weight: tp.Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Invokes the `BinaryCrossentropy` instance.

        Arguments:
            y_true: Ground truth values.
            y_pred: The predicted values.
            sample_weight: Acts as a
                coefficient for the loss. If a scalar is provided, then the loss is
                simply scaled by the given value. If `sample_weight` is a tensor of size
                `[batch_size]`, then the total loss for each sample of the batch is
                rescaled by the corresponding element in the `sample_weight` vector. If
                the shape of `sample_weight` is `[batch_size, d0, .. dN-1]` (or can be
                broadcasted to this shape), then each loss element of `y_pred` is scaled
                by the corresponding value of `sample_weight`. (Note on`dN-1`: all loss
                functions reduce by 1 dimension, usually axis=-1.)
        Returns:
            Loss values per sample.
        """

        return binary_cross_entropy(y_true, y_pred, from_logits=self._from_logits)
