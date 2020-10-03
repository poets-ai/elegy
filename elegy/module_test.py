from elegy.module import hooks_context
import inspect
from unittest import TestCase
import jax

import jax.numpy as jnp
import pytest

import elegy
from elegy import utils
import numpy as np


class ModuleTest(TestCase):
    class Linear(elegy.Module):
        def __init__(self, units):
            super().__init__()
            self.units = units

        def call(self, x):
            w = self.add_parameter("w", [x.shape[-1], self.units], initializer=jnp.ones)
            b = self.add_parameter("b", [self.units], initializer=jnp.ones)

            n = self.add_parameter(
                "n", [], dtype=jnp.int32, initializer=jnp.zeros, trainable=False
            )

            self.update_parameter("n", n + 1)

            y = jnp.dot(x, w) + b

            elegy.add_loss("activation_sum", jnp.sum(y))
            elegy.add_metric("activation_mean", jnp.mean(y))

            return y

    class MyModule(elegy.Module):
        def __init__(self):
            super().__init__()
            self.linear = ModuleTest.Linear(6)
            self.linear1 = ModuleTest.Linear(7)

        def call(self, x) -> np.ndarray:
            x = self.linear(x)
            x = self.linear1(x)
            self.bias = self.add_parameter("bias", [x.shape[-1]], jnp.float32, jnp.ones)
            return x + self.bias * 10

    def test_basic(self):
        x = np.random.uniform(-1, 1, size=(4, 5))
        m = ModuleTest.MyModule()
        m.init(x)
        y: jnp.ndarray = m(x)
        assert y.shape == (4, 7)
        print(m.get_parameters())

    def test_get_parameters(self):
        x = np.random.uniform(-1, 1, size=(4, 5))

        m = ModuleTest.MyModule()

        m.init(x)

        parameters = m.get_parameters()
        state = m.get_parameters(trainable=False)

        assert "bias" in parameters
        assert "linear" in parameters
        assert "w" in parameters["linear"]
        assert "b" in parameters["linear"]
        assert parameters["linear"]["n"] == 0
        assert parameters["linear1"]["n"] == 0
        assert "linear1" in parameters

        with elegy.hooks_context(summaries=True):
            y: jnp.ndarray = m(x)
            # y2: jnp.ndarray = m.call_jit(x)

            losses = elegy.get_losses()
            metrics = elegy.get_metrics()
            summaries = elegy.get_summaries()

        assert losses
        assert metrics
        assert summaries

        parameters = m.get_parameters()

        assert y.shape == (4, 7)
        assert "bias" in parameters
        assert "linear" in parameters
        assert "w" in parameters["linear"]
        assert "b" in parameters["linear"]
        assert m.linear.get_parameters()["n"] == 1
        assert parameters["linear"]["n"] == 1
        assert "linear1" in parameters

        assert "activation_sum_loss" in losses
        assert "my_module/linear/activation_mean" in metrics
        assert "my_module/linear_1/activation_mean" in metrics

        assert summaries[0][:2] == (m.linear, "my_module/linear")
        assert summaries[0][2].shape == (4, 6)
        assert summaries[1][:2] == (m.linear1, "my_module/linear_1")
        assert summaries[1][2].shape == (4, 7)
        assert summaries[2][:2] == (m, "my_module")
        assert summaries[2][2].shape == (4, 7)

        m.set_parameters(jax.tree_map(lambda x: -x, parameters))

        parameters = m.get_parameters()

        assert parameters["bias"][0] == -1
        assert m.linear.get_parameters()["w"][0, 0] == -1
        assert m.linear.get_parameters()["b"][0] == -1
        assert m.linear1.get_parameters()["w"][0, 0] == -1
        assert m.linear1.get_parameters()["b"][0] == -1

        assert m.parameters_size(include_submodules=False) == 7

        current_parameters = m.get_parameters()

        m.reset()

        parameters = m.get_parameters()

        assert jax.tree_leaves(parameters) == []
        assert m.parameters_size() == 0

        m.set_parameters(current_parameters)

        assert m.get_parameters()["bias"][0] == -1
        assert m.linear.get_parameters()["w"][0, 0] == -1
        assert m.linear.get_parameters()["b"][0] == -1
        assert m.linear1.get_parameters()["w"][0, 0] == -1
        assert m.linear1.get_parameters()["b"][0] == -1


class ModuleDynamicTest(TestCase):
    class Linear(elegy.Module):
        w: jnp.ndarray
        b: jnp.ndarray

        def __init__(self, units):
            super().__init__()
            self.units = units

        def call(self, x):
            w = self.add_parameter("w", [x.shape[-1], self.units], initializer=jnp.ones)
            b = self.add_parameter("b", [self.units], initializer=jnp.ones)

            elegy.next_rng_key()

            n = self.add_parameter(
                "n", [], dtype=jnp.int32, initializer=jnp.zeros, trainable=False
            )

            self.update_parameter("n", n + 1)

            y = jnp.dot(x, w) + b

            elegy.add_loss("activation_sum", jnp.sum(y))
            elegy.add_metric("activation_mean", jnp.mean(y))

            return y

    class MyModule(elegy.Module):
        linear: "ModuleDynamicTest.Linear"
        linear_1: "ModuleDynamicTest.Linear"

        def call(self, x) -> np.ndarray:
            x = ModuleDynamicTest.Linear(6)(x)
            x = ModuleDynamicTest.Linear(7)(x)
            self.bias = self.add_parameter("bias", [x.shape[-1]], initializer=jnp.ones)
            return x + self.bias * 10

    def test_basic(self):
        x = np.random.uniform(-1, 1, size=(4, 5))
        m = ModuleDynamicTest.MyModule()
        m.init(x)
        y: jnp.ndarray = m(x)
        assert y.shape == (4, 7)
        print(m.get_parameters)

    def test_get_parameters(self):
        x = np.random.uniform(-1, 1, size=(4, 5))
        initial_key = elegy.get_rng().key
        m = ModuleDynamicTest.MyModule()

        m.init_jit(x)

        assert not jnp.allclose(initial_key, elegy.get_rng().key)
        assert "bias" in m.get_parameters()
        assert "linear" in m.get_parameters()
        assert "w" in m.get_parameters()["linear"]
        assert "b" in m.get_parameters()["linear"]
        assert m.linear.get_parameters()["n"] == 0
        assert m.get_parameters()["linear"]["n"] == 0
        assert "linear_1" in m.get_parameters()

        with elegy.hooks_context(summaries=True):
            # y: jnp.ndarray = m(x)
            y: jnp.ndarray = m.jit(x)

            losses = elegy.get_losses()
            metrics = elegy.get_metrics()
            summaries = elegy.get_summaries()

        assert losses
        assert metrics
        assert summaries

        assert y.shape == (4, 7)
        assert "bias" in m.get_parameters()
        assert "linear" in m.get_parameters()
        assert "w" in m.get_parameters()["linear"]
        assert "b" in m.get_parameters()["linear"]
        assert m.linear.get_parameters()["n"] == 1
        assert m.get_parameters()["linear"]["n"] == 1
        assert "linear_1" in m.get_parameters()

        assert "activation_sum_loss" in losses
        assert "my_module/linear/activation_mean" in metrics
        assert "my_module/linear_1/activation_mean" in metrics

        assert summaries[0][:2] == (m.linear, "my_module/linear")
        assert summaries[0][2].shape == (4, 6)
        assert summaries[1][:2] == (m.linear_1, "my_module/linear_1")
        assert summaries[1][2].shape == (4, 7)
        assert summaries[2][:2] == (m, "my_module")
        assert summaries[2][2].shape == (4, 7)

        m.set_parameters(jax.tree_map(lambda x: -x, m.get_parameters()))

        assert m.get_parameters()["bias"][0] == -1
        assert m.linear.get_parameters()["w"][0, 0] == -1
        assert m.linear.get_parameters()["b"][0] == -1
        assert m.linear_1.get_parameters()["w"][0, 0] == -1
        assert m.linear_1.get_parameters()["b"][0] == -1

        assert m.parameters_size(include_submodules=False) == 7

        current_parameters = m.get_parameters()

        m.reset()

        assert jax.tree_leaves(m.get_parameters()) == []
        assert m.parameters_size() == 0

        m.set_parameters(current_parameters)

        assert m.get_parameters()["bias"][0] == -1
        assert m.linear.get_parameters()["w"][0, 0] == -1
        assert m.linear.get_parameters()["b"][0] == -1
        assert m.linear_1.get_parameters()["w"][0, 0] == -1
        assert m.linear_1.get_parameters()["b"][0] == -1

    def test_auto_init(self):
        x = np.random.uniform(-1, 1, size=(4, 5))
        initial_key = elegy.get_rng().key
        m = ModuleDynamicTest.MyModule()

        m(x)

        # THESE:
        assert m.linear.get_parameters()["n"] == 1
        assert m.get_parameters()["linear"]["n"] == 1

        assert not jnp.allclose(initial_key, elegy.get_rng().key)
        assert "bias" in m.get_parameters()
        assert "linear" in m.get_parameters()
        assert "w" in m.get_parameters()["linear"]
        assert "b" in m.get_parameters()["linear"]
        assert "linear_1" in m.get_parameters()

        with elegy.hooks_context(summaries=True):
            # y: jnp.ndarray = m(x)
            y: jnp.ndarray = m.jit(x)

            losses = elegy.get_losses()
            metrics = elegy.get_metrics()
            summaries = elegy.get_summaries()

        assert losses
        assert metrics
        assert summaries

        assert y.shape == (4, 7)
        assert "bias" in m.get_parameters()
        assert "linear" in m.get_parameters()
        assert "w" in m.get_parameters()["linear"]
        assert "b" in m.get_parameters()["linear"]
        assert m.linear.get_parameters()["n"] == 2
        assert m.get_parameters()["linear"]["n"] == 2
        assert "linear_1" in m.get_parameters()

        assert "activation_sum_loss" in losses
        assert "my_module/linear/activation_mean" in metrics
        assert "my_module/linear_1/activation_mean" in metrics

        assert summaries[0][:2] == (m.linear, "my_module/linear")
        assert summaries[0][2].shape == (4, 6)
        assert summaries[1][:2] == (m.linear_1, "my_module/linear_1")
        assert summaries[1][2].shape == (4, 7)
        assert summaries[2][:2] == (m, "my_module")
        assert summaries[2][2].shape == (4, 7)

        m.set_parameters(jax.tree_map(lambda x: -x, m.get_parameters()))

        assert m.get_parameters()["bias"][0] == -1
        assert m.linear.get_parameters()["w"][0, 0] == -1
        assert m.linear.get_parameters()["b"][0] == -1
        assert m.linear_1.get_parameters()["w"][0, 0] == -1
        assert m.linear_1.get_parameters()["b"][0] == -1

        assert m.parameters_size(include_submodules=False) == 7

        current_parameters = m.get_parameters()

        m.reset()

        assert jax.tree_leaves(m.get_parameters()) == []
        assert m.parameters_size() == 0

        m.set_parameters(current_parameters)

        assert m.get_parameters()["bias"][0] == -1
        assert m.linear.get_parameters()["w"][0, 0] == -1
        assert m.linear.get_parameters()["b"][0] == -1
        assert m.linear_1.get_parameters()["w"][0, 0] == -1
        assert m.linear_1.get_parameters()["b"][0] == -1


class TestTransforms(TestCase):
    def test_simple_0(self):

        total_called = 0

        @jax.jit
        def f(params, x):
            nonlocal total_called

            total_called += 1

            y = params["w"] * x + params["b"]
            params["w"] += 1.0

            return params, y

        params = {"w": jnp.array(1.0), "b": jnp.array(2.0)}
        params, outputs = f(params, 1)
        # m.set_parameters(params)

        print(params, type(params))

        assert total_called == 1

        params = {"b": params["b"], "w": params["w"]}
        params, outputs = f(params, 1)
        # m.set_parameters(params)

        assert total_called == 1

    def test_simple(self):

        total_called = 0

        class SomeModule(elegy.Module):
            n: jnp.ndarray

            def call(self, x):
                nonlocal total_called

                total_called += 1

                n = self.add_parameter("n", initializer=jnp.array(0))
                self.update_parameter("n", n + 1)

                if elegy.is_training():
                    return x + 1
                else:
                    return x - 1

        m = SomeModule()

        @jax.jit
        def f(params, x):

            m.set_parameters(params)

            outputs = m(x)

            return m.get_parameters(), outputs

        assert total_called == 0

        m.init(1)
        assert total_called == 1

        params = m.get_parameters()
        params, outputs = f(params, 1)
        m.set_parameters(params)

        assert total_called == 2

        params = m.get_parameters()
        print(params)
        params, outputs = f(params, 1)
        m.set_parameters(params)

        assert total_called == 2

    def test_jit(self):
        with elegy.training_context(True):
            total_called = 0

            class SomeModule(elegy.Module):
                n: jnp.ndarray

                def call(self, x):
                    nonlocal total_called

                    total_called += 1

                    n = self.add_parameter("n", initializer=jnp.array(0))
                    self.update_parameter("n", n + 1)

                    if elegy.is_training():
                        return x + 1
                    else:
                        return x - 1

            m = SomeModule()

            assert total_called == 0

            m.init(0)
            assert total_called == 1

            m_jit = elegy.jit(m)

            assert m.n == 0

            y = m_jit(0)

            assert y == 1
            assert m.n == 1
            assert total_called == 2

            y = m_jit(0)
            assert m.n == 2
            assert total_called == 2

            elegy.set_training(False)
            y = m_jit(0)
            assert y == -1
            assert total_called == 3
            assert m.n == 3

            with elegy.training_context(training=True):
                y = m_jit(0)
                assert y == 1
                assert total_called == 3
                assert m.n == 4

            with elegy.training_context(training=True), elegy.hooks_context():
                y = m_jit(0)
                assert y == 1
                assert total_called == 4
                assert m.n == 5

            with elegy.training_context(training=True), elegy.hooks_context():
                y = m_jit(0)
                assert y == 1
                assert total_called == 4
                assert m.n == 6

            with elegy.training_context(training=True), elegy.hooks_context(
                summaries=True
            ):
                y = m_jit(0)
                assert y == 1
                assert total_called == 5
                assert m.n == 7

            with elegy.training_context(training=False), elegy.hooks_context(
                summaries=True
            ):
                y = m_jit(0)
                assert y == -1
                assert total_called == 6
                assert m.n == 8

    def test_jit_auto_init(self):
        with elegy.training_context(True):
            total_called = 0

            class SomeModule(elegy.Module):
                n: jnp.ndarray

                def call(self, x):
                    nonlocal total_called

                    total_called += 1

                    n = self.add_parameter("n", initializer=jnp.array(0))
                    self.update_parameter("n", n + 1)

                    if elegy.is_training():
                        return x + 1
                    else:
                        return x - 1

            m = SomeModule()

            assert total_called == 0

            m.jit(0)
            assert total_called == 1

            m_jit = elegy.jit(m)

            assert m.n == 1

            y = m_jit(0)

            assert y == 1
            assert m.n == 2
            assert total_called == 2

            y = m_jit(0)
            assert m.n == 3
            assert total_called == 2

            elegy.set_training(False)
            y = m_jit(0)
            assert y == -1
            assert total_called == 3
            assert m.n == 4

            with elegy.training_context(training=True):
                y = m_jit(0)
                assert y == 1
                assert total_called == 3
                assert m.n == 5

            with elegy.training_context(training=True), elegy.hooks_context():
                y = m_jit(0)
                assert y == 1
                assert total_called == 4
                assert m.n == 6

            with elegy.training_context(training=True), elegy.hooks_context():
                y = m_jit(0)
                assert y == 1
                assert total_called == 4
                assert m.n == 7

            with elegy.training_context(training=True), elegy.hooks_context(
                summaries=True
            ):
                y = m_jit(0)
                assert y == 1
                assert total_called == 5
                assert m.n == 8

            with elegy.training_context(training=False), elegy.hooks_context(
                summaries=True
            ):
                y = m_jit(0)
                assert y == -1
                assert total_called == 6
                assert m.n == 9


class TestOthers(TestCase):
    def test_trainable(self):
        class SomeModule(elegy.Module):
            linear: elegy.nn.Linear

            def call(self, x):
                return elegy.nn.Linear(10)(x)

        x = np.random.uniform(-1, 1, size=(4, 5))
        m = SomeModule()
        m.init(x)

        params = m.get_parameters(trainable=True)
        assert "linear" in params
        assert "w" in params["linear"]
        assert "b" in params["linear"]

        m.linear.trainable = False

        params = m.get_parameters(trainable=True)
        assert "w" not in params["linear"]
        assert "b" not in params["linear"]

        params = m.get_parameters(trainable=False)
        assert "w" in params["linear"]
        assert "b" in params["linear"]

        m.trainable = True

        params = m.get_parameters(trainable=True)
        assert "w" in params["linear"]
        assert "b" in params["linear"]

        m.trainable = False

        params = m.get_parameters(trainable=True)
        assert "w" not in params["linear"]
        assert "b" not in params["linear"]

    def test_trainable_jit(self):
        total_called = 0

        class SomeModule(elegy.Module):
            linear: elegy.nn.Linear

            def call(self, x):
                nonlocal total_called
                total_called += 1
                return elegy.nn.Linear(10)(x)

        x = np.random.uniform(-1, 1, size=(4, 5))
        m = SomeModule()
        m_jit = elegy.jit(m)

        m.init(x)
        assert total_called == 1

        m_jit(x)
        assert total_called == 2

        m_jit(x)
        assert total_called == 2

        m.linear.trainable = False
        m_jit(x)
        assert total_called == 3

        m_jit(x)
        assert total_called == 3

        m.trainable = True
        m_jit(x)
        assert total_called == 3

        m.trainable = False
        m_jit(x)
        assert total_called == 4

        m_jit(x)
        assert total_called == 4

    def test_trainable_jit_method(self):
        total_called = 0

        class SomeModule(elegy.Module):
            linear: elegy.nn.Linear

            def call(self, x):
                nonlocal total_called
                total_called += 1
                return elegy.nn.Linear(10)(x)

        x = np.random.uniform(-1, 1, size=(4, 5))
        m = SomeModule()

        m.init_jit(x)
        assert total_called == 1

        m.jit(x)
        assert total_called == 2

        m.jit(x)
        assert total_called == 2

        m.linear.trainable = False
        m.jit(x)
        assert total_called == 3

        m.jit(x)
        assert total_called == 3

        m.trainable = True
        m.jit(x)
        assert total_called == 3

        m.trainable = False
        m.jit(x)
        assert total_called == 4

        m.jit(x)
        assert total_called == 4

    def test_module_system_docs(self):
        class Linear(elegy.Module):
            def __init__(self, n_out):
                super().__init__()
                self.n_out = n_out

            def call(self, x):
                w = self.add_parameter(
                    "w",
                    [x.shape[-1], self.n_out],
                    initializer=elegy.initializers.RandomUniform(),
                )
                b = self.add_parameter("b", [self.n_out], initializer=jnp.zeros)

                return jnp.dot(x, w) + b

        class MLP(elegy.Module):
            def call(self, x):
                x = Linear(64)(x)
                x = jax.nn.relu(x)
                x = Linear(32)(x)
                x = jax.nn.relu(x)
                x = Linear(1)(x)
                return x

        def loss_fn(x, y):
            y_pred = mlp(x)
            return jnp.mean(jnp.square(y - y_pred))

        def update(x, y):
            parameters = mlp.get_parameters(trainable=True)
            loss, gradients = elegy.value_and_grad(loss_fn, modules=mlp)(x, y)
            new_parameters = jax.tree_multimap(
                lambda p, g: p - 0.01 * g, parameters, gradients
            )
            mlp.set_parameters(new_parameters)

            return loss

        x = np.random.uniform(size=(15, 3))
        y = np.random.uniform(size=(15, 1))
        mlp = MLP()

        update_jit = elegy.jit(update, modules=mlp)

        for step in range(1):
            loss = update_jit(x, y)
