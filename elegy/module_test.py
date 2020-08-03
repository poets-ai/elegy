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
        w = self.get_parameter("w", [x.shape[-1], self.units], jnp.float32, jnp.ones)
        b = self.get_parameter("b", [self.units], jnp.float32, jnp.ones)

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
        self.bias = self.get_parameter("bias", [x.shape[-1]], jnp.float32, jnp.ones)
        return x + self.bias * 10


class ModuleTest(TestCase):
    def test_basic(self):
        x = np.random.uniform(-1, 1, size=(4, 5))
        module = MyModule()
        module.init()(x)
        y: np.ndarray
        y, context = module.apply()(x)
        assert y.shape == (4, 7)

    def test_get_parameters(self):
        x = np.random.uniform(-1, 1, size=(4, 5))

        m = MyModule()

        m.init()(x)

        assert "bias" in m.parameters
        assert "linear" in m.parameters
        assert "w" in m.parameters["linear"]
        assert "b" in m.parameters["linear"]
        assert m.linear.states["n"] == 0
        assert m.states["linear"]["n"] == 0
        assert "linear1" in m.parameters

        y: np.ndarray
        y, context = m.apply(get_summaries=True)(x)

        assert y.shape == (4, 7)
        assert "bias" in m.parameters
        assert "linear" in m.parameters
        assert "w" in m.parameters["linear"]
        assert "b" in m.parameters["linear"]
        assert m.linear.states["n"] == 1
        assert m.states["linear"]["n"] == 1
        assert "linear1" in m.parameters

        assert "activation_sum_loss" in context.losses
        assert "my_module/linear/activation_mean" in context.metrics
        assert "my_module/linear_1/activation_mean" in context.metrics

        assert context.summaries[0][:2] == (m.linear, "my_module/linear")
        assert context.summaries[0][2].shape == (4, 6)
        assert context.summaries[1][:2] == (m.linear1, "my_module/linear_1")
        assert context.summaries[1][2].shape == (4, 7)
        assert context.summaries[2][:2] == (m, "my_module")
        assert context.summaries[2][2].shape == (4, 7)

        m.parameters = jax.tree_map(lambda x: -x, m.parameters)

        assert m.parameters["bias"][0] == -1
        assert m.linear.parameters["w"][0, 0] == -1
        assert m.linear.parameters["b"][0] == -1
        assert m.linear1.parameters["w"][0, 0] == -1
        assert m.linear1.parameters["b"][0] == -1

        print(f"{m.parameters_size(include_submodules=False)=}")
        assert m.parameters_size(include_submodules=False) == 7

        print(f"{m.parameters_size()=}")
        m.clear_parameters()
        m.clear_states()

        assert m.parameters == {}
        assert m.parameters_size() == 0

        print(f"{m.parameters_size()=}")

        print(f"{inspect.signature(m.apply())=}")
        print(f"{inspect.signature(m)=}")

