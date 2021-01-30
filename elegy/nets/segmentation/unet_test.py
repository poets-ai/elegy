from elegy import utils

import jax.numpy as jnp
import numpy as np
from unittest import TestCase
import tempfile, os, pickle

import elegy


class UNetTest(TestCase):
    def test_basic_predict(self):
        model = elegy.Model(elegy.nets.unet.UNet_R18(), run_eagerly=True)
        x = np.ones((1, 64, 64, 3)).astype(np.float32) * 1e-12
        y = model.predict(x)
        assert jnp.all(y.shape == (1, 64, 64, 73))

        # test loading weights from file
        tempdir = tempfile.TemporaryDirectory()
        pklpath = os.path.join(tempdir.name, "delete_me.pkl")
        open(pklpath, "wb").write(pickle.dumps(model.module.get_parameters()))

        new_ur18 = elegy.nets.unet.UNet_R18(weights=pklpath)
        y2 = elegy.Model(new_ur18, run_eagerly=True).predict(x)

        assert np.allclose(y, y2)
