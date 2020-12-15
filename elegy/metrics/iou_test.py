import jax,jax.numpy as jnp
import numpy as np
import elegy
import unittest, pytest


from elegy.metrics.reduce_confusion_matrix import Reduction, ReduceConfusionMatrix
import tensorflow.keras as keras




class MeanIoUTest(unittest.TestCase):
    def test_basic(self):
        miou = elegy.metrics.MeanIoU()
        kmiou = keras.metrics.MeanIoU(num_classes=3)

        ytrue = jnp.array([0,0,1,1,2])
        ypred_sparse = jnp.array([0,1,1,1,0])
        ypred = jax.nn.one_hot(ypred_sparse, num_classes=3)

        iou0 = miou(ytrue, ypred)
        iou1 = kmiou(ytrue, ypred_sparse)

        assert np.allclose(iou0, iou1)


    def test_random(self):
        N_CLASSES = 10

        miou = elegy.metrics.MeanIoU()
        kmiou = keras.metrics.MeanIoU(num_classes=N_CLASSES)

        ytrue = np.random.randint(0,N_CLASSES, size=(4,100,100))
        ypred = np.random.random([4,100,100,10])

        iou0 = miou(ytrue, ypred)
        kmiou.update_state(ytrue, ypred.argmax(-1))
        iou1 = kmiou.result().numpy()
        assert np.allclose(iou0,iou1)

        #accumulate + sample weights
        ytrue = np.random.randint(0,N_CLASSES, size=(4,100,100))
        ypred = np.random.random([4,100,100,10])
        weight = np.random.random(ytrue.shape)

        iou0 = miou(ytrue, ypred, sample_weight=weight)
        kmiou.update_state(ytrue, ypred.argmax(-1), sample_weight=weight)
        iou1 = kmiou.result().numpy()
        assert np.allclose(iou0,iou1)

    def test_selected_classes(self):
        #only considering classes 0 and 4
        miou = elegy.metrics.MeanIoU(classes = [0,4])

        ytrue        = jnp.array([0,0,1,1,2,4,4,4])
        ypred_sparse = jnp.array([0,1,1,1,0,4,4,0])
        ypred = jax.nn.one_hot(ypred_sparse, num_classes=5)

        iou0 = miou(ytrue, ypred)
        assert jnp.allclose(iou0, (1/4 + 2/3)/2 )

        #test ignore_index
        miou = elegy.metrics.MeanIoU(classes = [0,4], ignore_index=2)
        iou1 = miou(ytrue, ypred)
        assert jnp.allclose(iou1, (1/3 + 2/3)/2 )

        





class MultiClassReductionsTest(unittest.TestCase):
    def test_mc_tp(self):
        with pytest.raises(ValueError):
            mc_tp = ReduceConfusionMatrix(Reduction.MULTICLASS_TRUE_POSITIVES)
        
        mc_tp = ReduceConfusionMatrix(Reduction.MULTICLASS_TRUE_POSITIVES, n_classes=5)

        ytrue = jnp.array([0,0,0,1,1,1,2,3])
        ypred = jnp.array([1,0,0,0,0,1,2,2])

        tp0 = mc_tp(ytrue, ypred)
        assert all(tp0 == jnp.array([2,1,1,0,0]))
        
        #accumulate
        ypred = ytrue.copy()
        tp1 = mc_tp(ytrue, ypred)
        assert all(tp1 == jnp.array([5,4,2,1,0]))

        #randomized
        ytrue = np.random.randint(0,10, size=(100,100))
        ypred = np.random.randint(0,10, size=(100,100))
        mc_tp = ReduceConfusionMatrix(Reduction.MULTICLASS_TRUE_POSITIVES, n_classes=10)
        tp2 = mc_tp(ytrue, ypred)
        assert tp2.sum() == (ytrue==ypred).sum()
    
    def test_mc_fp(self):
        mc_fp = ReduceConfusionMatrix(Reduction.MULTICLASS_FALSE_POSITIVES, n_classes=5)

        ytrue = jnp.array([0,0,0,1,1,1,2,3])
        ypred = jnp.array([1,0,0,0,0,1,2,2])

        fp0 = mc_fp(ytrue, ypred)
        assert all(fp0 == jnp.array([2,1,1,0,0]))

        ypred = ytrue.copy()
        fp1 = mc_fp(ytrue, ypred)
        assert all(fp0 == jnp.array([2,1,1,0,0])) #no new false positives

    def test_mc_fn(self):
        mc_fn = ReduceConfusionMatrix(Reduction.MULTICLASS_FALSE_NEGATIVES, n_classes=5)

        ytrue = jnp.array([0,0,0,1,1,1,2,3])
        ypred = jnp.array([1,0,0,0,0,1,2,2])

        fp0 = mc_fn(ytrue, ypred)
        assert all(fp0 == jnp.array([1,2,0,1,0]))

        ypred = ytrue.copy()
        fp1 = mc_fn(ytrue, ypred)
        assert all(fp0 == jnp.array([1,2,0,1,0])) #no new false negatives


