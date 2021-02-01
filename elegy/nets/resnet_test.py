from elegy import utils

import jax, jax.numpy as jnp
import numpy as np
from unittest import TestCase
import tempfile, os, pickle
import PIL, urllib

import elegy


class ResNetTest(TestCase):
    def test_basic_predict(self):
        # FIXME: test succeeds if run alone or if run on the cpu-only version of jax
        # test fails with "DNN library is not found" if run on gpu with all other tests together

        model = elegy.Model(elegy.nets.resnet.ResNet18(), run_eagerly=True)
        assert isinstance(model.module, elegy.Module)

        x = np.random.random((2, 224, 224, 3)).astype(np.float32)
        y = model.predict(x)

        # update_modules results in a call to `set_default_parameters` for elegy Modules
        # it might be better to have the user call this explicitly to avoid potential OOM
        model.update_modules()

        assert jnp.all(y.shape == (2, 1000))

        # test loading weights from file
        with tempfile.TemporaryDirectory() as tempdir:
            pklpath = os.path.join(tempdir, "delete_me.pkl")
            open(pklpath, "wb").write(
                pickle.dumps(model.module.get_default_parameters())
            )

            new_r18 = elegy.nets.resnet.ResNet18(weights=pklpath)
            y2 = elegy.Model(new_r18, run_eagerly=True).predict(x)

        assert np.allclose(y, y2, rtol=0.001)

    def test_autodownload_pretrained_r18(self):
        fname, _ = urllib.request.urlretrieve(
            "https://upload.wikimedia.org/wikipedia/commons/e/e4/A_French_Bulldog.jpg"
        )
        im = np.array(PIL.Image.open(fname).resize([224, 224])) / np.float32(255)

        r18 = elegy.nets.resnet.ResNet18(weights="imagenet")
        with jax.disable_jit():
            assert elegy.Model(r18).predict(im[np.newaxis]).argmax() == 245

    def test_autodownload_pretrained_r50(self):
        fname, _ = urllib.request.urlretrieve(
            "https://upload.wikimedia.org/wikipedia/commons/e/e4/A_French_Bulldog.jpg"
        )
        im = np.array(PIL.Image.open(fname).resize([224, 224])) / np.float32(255)

        r50 = elegy.nets.resnet.ResNet50(weights="imagenet")
        with jax.disable_jit():
            assert elegy.Model(r50).predict(im[np.newaxis]).argmax() == 245
