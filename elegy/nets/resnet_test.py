from elegy import utils

import jax.numpy as jnp
from unittest import TestCase

import elegy


class ResNetTest(TestCase):
    def test_basic_predict(self):
        # FIXME: test succeeds if run alone or if run on the cpu-only version of jax
        # test fails with "DNN library is not found" if run on gpu with all other tests together

        model = elegy.Model(elegy.nets.resnet.ResNet18())
        y = model.predict(jnp.zeros((2, 224, 224, 3)))
        assert jnp.all(y.shape == (2, 1000))
