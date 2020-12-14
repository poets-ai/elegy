import jax, jax.numpy as jnp
import typing as tp
from elegy import types

from elegy.metrics.reduce_confusion_matrix import Reduction, ReduceConfusionMatrix
from elegy.metrics.metric import Metric


def iou(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    sample_weight: jnp.ndarray,
    true_positives: ReduceConfusionMatrix,
    false_positives: ReduceConfusionMatrix,
    false_negatives: ReduceConfusionMatrix,
) -> jnp.ndarray:
    '''Computes the Intersection-over-Union or Jaccard index. Class-agnostic.'''
    
    tp = true_positives(y_true, y_pred, sample_weight)
    fp = false_positives(y_true, y_pred, sample_weight)
    fn = false_negatives(y_true, y_pred, sample_weight)

    iou = tp / (fp + fn + tp)
    return iou





class MeanIoU(Metric):
    def __init__(self, on: tp.Optional[types.IndexLike] = None, classes:tp.Optional[jnp.ndarray] = None, **kwargs):
        super().__init__(on=on, **kwargs)
        self._initialized = False
        self._classes = classes
    
    def call(
        self,
        y_true: jnp.ndarray,
        y_pred: jnp.ndarray,
        sample_weight: tp.Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        
        if not self._initialized:
            n = y_pred.shape[-1]
            self.TP = ReduceConfusionMatrix(Reduction.MULTICLASS_TRUE_POSITIVES, n_classes=n)
            self.FP = ReduceConfusionMatrix(Reduction.MULTICLASS_FALSE_POSITIVES, n_classes=n)
            self.FN = ReduceConfusionMatrix(Reduction.MULTICLASS_FALSE_NEGATIVES, n_classes=n)
            if self._classes is None:
                self._classes = jnp.arange(n)
            self._initialized = True
        
        y_pred_sparse = jnp.argmax(y_pred,axis=-1)
        per_class_iou = iou(y_true, y_pred_sparse, sample_weight, self.TP, self.FP, self.FN)
        selected_iou = per_class_iou[self._classes]
        miou = jnp.mean(selected_iou)
        return miou


