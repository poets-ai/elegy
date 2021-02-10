from unittest import TestCase

import numpy as np

import elegy


class LinearTest(TestCase):
    def test_connects(self):

        x = np.random.uniform(-1, 1, size=(4, 3))
        linear = elegy.nn.Linear(5)

        y_pred = linear.call_with_defaults(rng=elegy.RNGSeq(42))(x)

        assert y_pred.shape == (4, 5)

    def test_on_model(self):

        model = elegy.Model(module=elegy.nn.Linear(2))

        x = np.ones([3, 5])

        y_pred = model.predict(x)
        logs = model.evaluate(x)
