from elegy import utils

import jax.numpy as jnp
import numpy as np
from unittest import TestCase
import tempfile, os, pickle

import elegy


class ResNetTest(TestCase):
    def test_basic_predict(self):
        # FIXME: test succeeds if run alone or if run on the cpu-only version of jax
        # test fails with "DNN library is not found" if run on gpu with all other tests together

        model = elegy.Model(elegy.nets.resnet.ResNet18(), run_eagerly=True)
        x = np.random.random((2, 224, 224, 3)).astype(np.float32)
        y = model.predict(x)
        assert jnp.all(y.shape == (2, 1000))

        # test loading weights from file
        tempdir = tempfile.TemporaryDirectory()
        pklpath = os.path.join(tempdir.name, "delete_me.pkl")
        open(pklpath, "wb").write(pickle.dumps(model.module.get_parameters()))

        new_r18 = elegy.nets.resnet.ResNet18(weights=pklpath)
        y2 = elegy.Model(new_r18, run_eagerly=True).predict(x)

        assert np.allclose(y, y2)
