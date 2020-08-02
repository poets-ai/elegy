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
        w = self.get_parameter("w", [x.shape[-1], self.units], jnp.float32, jnp.ones)
        b = self.get_parameter("b", [self.units], jnp.float32, jnp.ones)
        return jnp.dot(x, w) + b


class MyModule(elegy.Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(7)

    def call(self, x) -> np.ndarray:
        x = self.linear(x)
        self.bias = self.get_parameter("bias", [x.shape[-1]], jnp.float32, jnp.ones)
        return x + self.bias * 10


class ModuleTest(TestCase):
    def test_basic(self):
        x = np.random.uniform(-1, 1, size=(4, 5))
        module = MyModule()
        module.init()(x)
        y = module.apply()(x)
        assert y.shape == (4, 7)

    def test_get_parameters(self):
        x = np.random.uniform(-1, 1, size=(4, 5))

        m = MyModule()

        # m.init()(x)

        # assert m.linear.n == 0
        # assert m.states["linear"]["n"] == 0

        context: elegy.ApplyOutput
        context = m.apply(get_summaries=False, return_context=True)(x)

        y: np.ndarray = context[0]

        # assert m.linear.n == 1
        # assert m.states["linear"]["n"] == 1

        assert y.shape == (4, 7)

        m.parameters = jax.tree_map(lambda x: -x, m.parameters)

        print(m.parameters)
        print(jax.tree_map(lambda x: x.shape, m.parameters))

        print(context)

