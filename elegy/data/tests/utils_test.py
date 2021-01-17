import math
from unittest import TestCase

import numpy as np
import jax.numpy as jnp

import pytest

from elegy.data import utils


class TrainValidationSplitTest(TestCase):
    def test_basic(self):
        x_all = np.random.uniform(size=(100, 32, 32, 3))
        y_all = np.random.uniform(size=(100, 1))
        sample_weight_all = None
        split = 0.2

        (x, y, sample_weight), validation_data = utils.train_validation_split(
            (x_all, y_all, sample_weight_all), validation_split=0.2, shuffle=False
        )

        assert x.shape[0] == int(x_all.shape[0] * (1 - split))
        assert y.shape[0] == int(y_all.shape[0] * (1 - split))
        assert sample_weight is None

        (x, y, sample_weight) = validation_data
        assert x.shape[0] == int(x_all.shape[0] * split)
        assert y.shape[0] == int(y_all.shape[0] * split)
        assert sample_weight is None
