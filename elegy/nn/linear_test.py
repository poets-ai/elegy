from unittest import TestCase

import numpy as np

import elegy


class LinearTest(TestCase):
    def test_connects(self):

        x = np.random.uniform(-1, 1, size=(4, 3))
        linear = elegy.nn.Linear(5)

        linear(x)
        y_pred = linear(x)

        assert y_pred.shape == (4, 5)
