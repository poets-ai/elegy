import inspect
from unittest import TestCase
import jax

import jax.numpy as jnp
import pytest

import elegy
from elegy import utils
import numpy as np


class Linear(elegy.Module):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def call(self, x):
        w = elegy.get_parameter("w", [x.shape[-1], self.units], jnp.float32, jnp.ones)
        b = elegy.get_parameter("b", [self.units], jnp.float32, jnp.ones)

        n = self.get_state("n", [], np.int32, jnp.zeros)

        self.set_state("n", n + 1)

        y = jnp.dot(x, w) + b

        elegy.add_loss("activation_sum", jnp.sum(y))
        elegy.add_metric("activation_mean", jnp.mean(y))

        return y


class MyModule(elegy.Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(6)
        self.linear1 = Linear(7)

    def call(self, x) -> np.ndarray:
        x = self.linear(x)
        x = self.linear1(x)
        self.bias = elegy.get_parameter("bias", [x.shape[-1]], jnp.float32, jnp.ones)
        return x + self.bias * 10


class Count(elegy.Module):
    def call(self):

        n = self.get_state("n", [], np.int32, jnp.zeros)
        n += 1
        self.set_state("n", n)

        return 1.0 / n


# class ModuleTest(TestCase):
#     def test_basic(self):

#     def test_get_parameters(self):
#         x = np.random.uniform(-1, 1, size=(4, 5))

#         module = MyModule()
#         model = elegy.Model(module)
