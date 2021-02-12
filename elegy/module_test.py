from elegy import types
from unittest import TestCase

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import elegy


class NewModuleTest(TestCase):
    def test_init_or_apply(self):
        class M(elegy.Module):
            def call(self, x):
                m = self.add_parameter("m", lambda: 1)
                n = self.add_parameter("n", lambda: 2)

                self.update_parameter("m", m + 1)
                self.update_parameter("n", n + 1)
                return x + 1

        parameters = None
        collections = None

        m = M()
        y_pred, parameters, collections = m.init_or_apply(parameters, collections)(2.0)

        assert y_pred == 3
        assert parameters == {"m": 1, "n": 2}
        assert collections == {}

        y_pred, parameters, collections = m.init_or_apply(parameters, collections)(
            y_pred
        )

        assert y_pred == 4
        assert parameters == {"m": 2, "n": 3}
        assert collections == {}

    def test_set_parameters(self):
        class M(elegy.Module):
            def call(self, x):
                self.add_parameter("m", lambda: 1)
                self.add_parameter("n", lambda: 2)
                return x

        m = M()
        m.init()(2.0)

        parameters = {"m": 1, "n": 10}
        collections = {}

        m.set_default_parameters(parameters, collections)

        assert m.m == 1
        assert m.n == 10

        with pytest.raises(ValueError):
            parameters = {"m": 100}
            collections = {
                "xyz": {"n": 100},
            }
            m.set_default_parameters(parameters, collections)

        assert m.m == 1
        assert m.n == 10

    def test_get_parameters(self):
        class M(elegy.Module):
            def call(self, x):
                self.add_parameter("m", lambda: 1)
                self.add_parameter("n", lambda: 2)
                return x

        m = M()
        m.init()(2.0)

        collections = {"parameters": {"m": 1, "n": 10}}
        paramters = collections.pop("parameters")

        m.set_default_parameters(paramters, collections)

        parameters, collections = m.get_default_parameters()

        assert parameters == {"m": 1, "n": 10}
        assert collections == {}

    def test_basic(self):
        class M(elegy.Module):
            def call(self, x):
                n = self.add_parameter("n", lambda: 1)
                return n + x

        m = M()

        y, parameters, collections = m.init()(2.0)

        assert parameters == {"n": 1}
        assert collections == {}
        assert y == 3

    def test_basic_call_error(self):
        class M(elegy.Module):
            def call(self, x):
                n = self.add_parameter("n", lambda: 1)
                return n + x

        m = M()

        with pytest.raises(types.NoContext):
            y = m(2.0)

    def test_basic_call(self):
        class M(elegy.Module):
            def call(self, x):
                n = self.add_parameter("n", lambda: 1)
                return n + x

        m = M()

        y = m.call_with_defaults()(2.0)
        parameters, collections = m.get_default_parameters()

        assert y == 3
        assert parameters == {"n": 1}
        assert collections == {}

        # update params
        m.set_default_parameters({"n": 20}, {})
        y = m.call_with_defaults()(2.0)
        parameters, collections = m.get_default_parameters()

        assert y == 22
        assert parameters == {"n": 20}
        assert collections == {}

    def test_basic_apply(self):
        class M(elegy.Module):
            def call(self, x):
                n = self.add_parameter("n", lambda: 1)
                return n + x

        m = M()

        with pytest.raises(ValueError):
            m.get_default_parameters()

        y, parameters, collections = m.init()(2.0)

        assert y == 3
        assert parameters == {"n": 1}
        assert collections == {}

        # set default parameters
        m.set_default_parameters(parameters, collections)

        # run with new params
        y, parameters, collections = m.apply({"n": 5}, {})(2.0)

        assert y == 7
        assert parameters == {"n": 5}
        assert collections == {}

        # check defaults are not modify by apply.
        default_params, default_collections = m.get_default_parameters()
        assert default_params == {"n": 1}
        assert default_collections == {}

    def test_basic_compose(self):
        class A(elegy.Module):
            def call(self, x):
                b = self.add_parameter("b", lambda: 10)
                return b + x

        class M(elegy.Module):
            def call(self, x):
                n = self.add_parameter("n", lambda: 1, trainable=False)
                x = A()(x)
                return n + x

        m = M()

        y, parameters, collections = m.init()(2.0)

        assert y == 13
        assert parameters == {"a": {"b": 10}}
        assert collections == {"states": {"n": 1, "a": {}}}

    def test_basic_list_submodules(self):
        class A(elegy.Module):
            def call(self, x):
                b = self.add_parameter("b", lambda: 10)
                return b + x

        class M(elegy.Module):
            def __init__(self):
                super().__init__()
                self.ais = [A(), A()]

            def call(self, x):
                n = self.add_parameter("n", lambda: 1)
                x = self.ais[1](x)
                return n + x

        m = M()

        with elegy.hooks.context(set_all=True):
            y, parameters, collections = m.init()(2.0)
            summaries = elegy.hooks.get_summaries()

        assert summaries == [
            (
                ("ais/1",),
                m.ais[1],
                12,
            ),
            (
                (),
                m,
                13,
            ),
        ]
        assert parameters == {
            "n": 1,
            "ais/0": {},
            "ais/1": {"b": 10},
        }
        assert collections == {}
        assert y == 13

    def test_basic_dynamic_submodules(self):
        class A(elegy.Module):
            def call(self, x):
                b = self.add_parameter("b", lambda: 10)
                return b + x

        class M(elegy.Module):
            a: A
            a_1: A

            def call(self, x):
                ais = [A(), A()]
                n = self.add_parameter("n", lambda: 1)
                x = ais[1](x)
                return n + x

        m = M()

        with elegy.hooks.context(set_all=True):
            y, params, collections = m.init()(2.0)
            summaries = elegy.hooks.get_summaries()

        assert summaries == [
            (
                ("a_1",),
                m.a_1,
                12,
            ),
            (
                (),
                m,
                13,
            ),
        ]
        assert params == {
            "n": 1,
            "a": {},
            "a_1": {
                "b": 10,
            },
        }
        assert collections == {}
        assert y == 13

    def test_basic_update(self):
        class M(elegy.Module):
            def call(self, x):
                n = self.add_parameter("n", lambda: 1)
                self.update_parameter("n", n + 1)
                return n + x

        m = M()

        y, parameters, collections = m.init()(2.0)

        assert y == 3
        assert parameters == {"n": 1}
        assert collections == {}

        y, parameters, collections = m.apply(parameters, collections)(2.0)
        assert y == 3

        y, parameters, collections = m.apply(parameters, collections)(2.0)
        assert y == 4

        # jit has to use functional API
        y, parameters, collections = m.apply_jit(parameters, collections)(2.0)
        assert y == 5

        # error is raised if no parameters are given
        with pytest.raises(TypeError):
            y, parameters, collections = m.apply_jit()(2.0)


class ModuleTest(TestCase):
    class Linear(elegy.Module):
        def __init__(self, units):
            super().__init__()
            self.units = units

        def call(self, x):
            w = self.add_parameter("w", lambda: jnp.ones([x.shape[-1], self.units]))
            b = self.add_parameter("b", lambda: jnp.ones([self.units]))
            n = self.add_parameter("n", lambda: jnp.zeros([]), trainable=False)

            self.update_parameter("n", n + 1)

            y = jnp.dot(x, w) + b

            path = elegy.module.get_module_path_str(self)
            elegy.hooks.add_loss(f"activation_sum", jnp.sum(y))
            elegy.hooks.add_metric(f"{path}/activation_mean", jnp.mean(y))

            return y

    class MyModule(elegy.Module):
        def __init__(self):
            super().__init__()
            self.linear = ModuleTest.Linear(6)
            self.linear1 = ModuleTest.Linear(7)

        def call(self, x) -> np.ndarray:
            x = self.linear(x)
            x = self.linear1(x)
            self.bias = self.add_parameter(
                "bias", lambda: jnp.ones([x.shape[-1]], jnp.float32)
            )
            return x + self.bias * 10

    def test_basic(self):
        x = np.random.uniform(-1, 1, size=(4, 5))
        m = ModuleTest.MyModule()
        y = m.call_with_defaults()(x)
        assert y.shape == (4, 7)

    def test_get_parameters(self):
        x = np.random.uniform(-1, 1, size=(4, 5))

        m = ModuleTest.MyModule()

        y, parameters, collections = m.init()(x)

        states = collections["states"]

        assert "bias" in parameters
        assert "linear" in parameters
        assert "w" in parameters["linear"]
        assert "b" in parameters["linear"]
        assert states["linear"]["n"] == 0
        assert states["linear1"]["n"] == 0
        assert "linear1" in parameters

        with elegy.hooks.context(set_all=True):
            y, parameters, collections = m.apply(parameters, collections)(x)
            # y2: jnp.ndarray = m.call_jit(x)

            losses = elegy.hooks.get_losses()
            metrics = elegy.hooks.get_metrics()
            summaries = elegy.hooks.get_summaries()

        assert losses
        assert metrics
        assert summaries

        states = collections["states"]

        assert y.shape == (4, 7)
        assert "bias" in parameters
        assert "linear" in parameters
        assert "w" in parameters["linear"]
        assert "b" in parameters["linear"]
        assert states["linear"]["n"] == 1
        assert "linear1" in parameters

        assert "activation_sum_loss" in losses
        assert "linear/activation_mean" in metrics
        assert "linear1/activation_mean" in metrics

        assert summaries[0][:2] == (("linear",), m.linear)
        assert summaries[0][2].shape == (4, 6)
        assert summaries[1][:2] == (("linear1",), m.linear1)
        assert summaries[1][2].shape == (4, 7)
        assert summaries[2][:2] == ((), m)
        assert summaries[2][2].shape == (4, 7)

        m.set_default_parameters(
            jax.tree_map(lambda x: -x, parameters),
            jax.tree_map(lambda x: -x, collections),
        )

        parameters, collections = m.get_default_parameters()
        states = collections["states"]

        assert parameters["bias"][0] == -1
        assert m.linear.get_default_parameters()[0]["w"][0, 0] == -1
        assert m.linear.get_default_parameters()[0]["b"][0] == -1
        assert m.linear1.get_default_parameters()[0]["w"][0, 0] == -1
        assert m.linear1.get_default_parameters()[0]["b"][0] == -1

        current_parameters, current_collection = m.get_default_parameters()

        m.clear_default_parameters()

        parameters, collections = m.get_default_parameters()

        assert parameters is None
        assert jax.tree_leaves(collections) == []
        assert elegy.utils.parameters_count(collections) == 0

        m.set_default_parameters(current_parameters, current_collection)

        assert m.get_default_parameters()[0]["bias"][0] == -1
        assert m.linear.get_default_parameters()[0]["w"][0, 0] == -1
        assert m.linear.get_default_parameters()[0]["b"][0] == -1
        assert m.linear1.get_default_parameters()[0]["w"][0, 0] == -1
        assert m.linear1.get_default_parameters()[0]["b"][0] == -1

    def test_set_parameters_shape_check(self):
        x = np.random.uniform(-1, 1, size=(4, 5))
        m = ModuleTest.MyModule()

        m.init(set_defaults=True)(x)

        params0, collections0 = m.get_default_parameters()
        # new random params
        params1 = jax.tree_map(lambda x: np.random.random(x.shape), params0)
        collections1 = jax.tree_map(lambda x: np.random.random(x.shape), collections0)
        # set a parameter with incorrect shape
        params1["linear1"]["w"] = np.zeros([10, 10])

        # should raise error when trying to set the incorrect params
        with pytest.raises(ValueError):
            m.set_default_parameters(params1, collections1)

        # the parameters should not have changed despite the error
        assert np.all(
            jax.tree_leaves(
                jax.tree_multimap(
                    lambda x, y: np.allclose(x, y),
                    params0,
                    m.get_default_parameters()[0],
                )
            )
        )

        # should not raise an error
        with pytest.raises(ValueError):
            m.set_default_parameters(params1, collections1)

        # linear1.w should not change
        assert np.allclose(
            m.get_default_parameters()[0]["linear1"]["w"],
            params0["linear1"]["w"],
        )
        # but all others
        assert np.all(
            jax.tree_multimap(
                lambda x, y: np.allclose(x, y) if x.shape == y.shape else True,
                params1,
                m.get_default_parameters()[0],
            )
        )

        # remove a parameter
        params1["linear1"].pop("w")
        # should raise with check_missing=True
        with pytest.raises(ValueError):
            m.set_default_parameters(params1, collections1)


class ModuleDynamicTest(TestCase):
    class Linear(elegy.Module):
        w: jnp.ndarray
        b: jnp.ndarray

        def __init__(self, units):
            super().__init__()
            self.units = units

        def call(self, x):
            w = self.add_parameter("w", lambda: jnp.ones([x.shape[-1], self.units]))
            b = self.add_parameter("b", lambda: jnp.ones([self.units]))
            n = self.add_parameter("n", lambda: jnp.zeros([]), trainable=False)

            self.update_parameter("n", n + 1)

            y = jnp.dot(x, w) + b

            path = elegy.module.get_module_path_str(self)
            elegy.hooks.add_loss(f"activation_sum", jnp.sum(y))
            elegy.hooks.add_metric(f"{path}/activation_mean", jnp.mean(y))

            return y

    class MyModule(elegy.Module):
        linear: "ModuleDynamicTest.Linear"
        linear_1: "ModuleDynamicTest.Linear"

        def call(self, x) -> np.ndarray:
            x = ModuleDynamicTest.Linear(6)(x)
            x = ModuleDynamicTest.Linear(7)(x)
            self.bias = self.add_parameter("bias", lambda: jnp.ones([x.shape[-1]]))
            return x + self.bias * 10

    def test_basic(self):
        x = np.random.uniform(-1, 1, size=(4, 5))
        m = ModuleDynamicTest.MyModule()
        y: jnp.ndarray = m.call_with_defaults()(x)
        assert y.shape == (4, 7)
        print(m.get_default_parameters)

    def test_get_parameters(self):
        x = np.random.uniform(-1, 1, size=(4, 5))
        m = ModuleDynamicTest.MyModule()

        m.init_jit(set_defaults=True)(x)

        parameters, collections = m.get_default_parameters()
        states = collections["states"]

        assert "bias" in parameters
        assert "linear" in parameters
        assert "w" in parameters["linear"]
        assert "b" in parameters["linear"]
        assert m.linear.get_default_parameters()[1]["states"]["n"] == 0
        assert states["linear"]["n"] == 0
        assert "linear_1" in parameters

        with elegy.hooks.context(set_all=True):
            # y: jnp.ndarray = m.call_with_defaults()(x)
            parameters, collections = m.get_default_parameters()
            y: jnp.ndarray
            y, parameters, collections = m.apply_jit(parameters, collections)(x)
            m.set_default_parameters(parameters, collections)

            states = collections["states"]

            losses = elegy.hooks.get_losses()
            metrics = elegy.hooks.get_metrics()
            summaries = elegy.hooks.get_summaries()

        assert losses is not None
        assert metrics
        assert summaries

        assert y.shape == (4, 7)
        assert "bias" in parameters
        assert "linear" in parameters
        assert "w" in parameters["linear"]
        assert "b" in parameters["linear"]
        assert m.linear.get_default_parameters()[1]["states"]["n"] == 1
        assert states["linear"]["n"] == 1
        assert "linear_1" in parameters

        assert "activation_sum_loss" in losses
        assert "linear/activation_mean" in metrics
        assert "linear_1/activation_mean" in metrics

        assert summaries[0][:2] == (("linear",), m.linear)
        assert summaries[0][2].shape == (4, 6)
        assert summaries[1][:2] == (("linear_1",), m.linear_1)
        assert summaries[1][2].shape == (4, 7)
        assert summaries[2][:2] == ((), m)
        assert summaries[2][2].shape == (4, 7)

        m.set_default_parameters(
            jax.tree_map(lambda x: -x, m.get_default_parameters()[0]),
            jax.tree_map(lambda x: -x, m.get_default_parameters()[1]),
        )

        parameters, collections = m.get_default_parameters()
        states = collections["states"]

        assert parameters["bias"][0] == -1
        assert m.linear.get_default_parameters()[0]["w"][0, 0] == -1
        assert m.linear.get_default_parameters()[0]["b"][0] == -1
        assert m.linear_1.get_default_parameters()[0]["w"][0, 0] == -1
        assert m.linear_1.get_default_parameters()[0]["b"][0] == -1

        current_parameters, current_collections = m.get_default_parameters()

        m.clear_default_parameters()

        assert m.get_default_parameters()[0] is None
        assert jax.tree_leaves(m.get_default_parameters()[1]) == []
        assert elegy.utils.parameters_count(m.get_default_parameters()[1]) == 0

        m.set_default_parameters(current_parameters, current_collections)

        assert m.get_default_parameters()[0]["bias"][0] == -1
        assert m.linear.get_default_parameters()[0]["w"][0, 0] == -1
        assert m.linear.get_default_parameters()[0]["b"][0] == -1
        assert m.linear_1.get_default_parameters()[0]["w"][0, 0] == -1
        assert m.linear_1.get_default_parameters()[0]["b"][0] == -1

    def test_auto_init(self):
        x = np.random.uniform(-1, 1, size=(4, 5))
        m = ModuleDynamicTest.MyModule()

        m.call_with_defaults()(x)

        # THESE:
        assert m.linear.get_default_parameters()[1]["states"]["n"] == 1
        assert m.get_default_parameters()[1]["states"]["linear"]["n"] == 1

        assert "bias" in m.get_default_parameters()[0]
        assert "linear" in m.get_default_parameters()[0]
        assert "w" in m.get_default_parameters()[0]["linear"]
        assert "b" in m.get_default_parameters()[0]["linear"]
        assert "linear_1" in m.get_default_parameters()[0]


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
        # m.set_default_parameters(params)

        print(params, type(params))

        assert total_called == 1

        params = {"b": params["b"], "w": params["w"]}
        params, outputs = f(params, 1)
        # m.set_default_parameters(params)

        assert total_called == 1

    def test_simple_defaults(self):

        total_called = 0

        class SomeModule(elegy.Module):
            n: jnp.ndarray

            def call(self, x):
                nonlocal total_called

                total_called += 1

                n = self.add_parameter("n", lambda: jnp.array(0))
                self.update_parameter("n", n + 1)

                if self.is_training():
                    return x + 1
                else:
                    return x - 1

        m = SomeModule()

        @jax.jit
        def f(params, collections, x):

            m.set_default_parameters(params, collections)

            outputs = m.call_with_defaults()(x)

            params, collections = m.get_default_parameters()

            return params, collections, outputs

        assert total_called == 0

        _, parameters, collections = m.init(set_defaults=True)(1)
        assert total_called == 1

        parameters, collections, outputs = f(parameters, collections, 1)
        m.set_default_parameters(parameters, collections)

        assert total_called == 2

        parameters, collections = m.get_default_parameters()
        print(collections)
        parameters, collections, outputs = f(parameters, collections, 1)
        m.set_default_parameters(parameters, collections)

        assert total_called == 2

    def test_simple_apply(self):

        total_called = 0

        class SomeModule(elegy.Module):
            n: jnp.ndarray

            def call(self, x):
                nonlocal total_called

                total_called += 1

                n = self.add_parameter("n", lambda: jnp.array(0))
                self.update_parameter("n", n + 1)

                if self.is_training():
                    return x + 1
                else:
                    return x - 1

        m = SomeModule()

        @jax.jit
        def f(parameters, collections, x):

            y, parameters, collections = m.apply(parameters, collections)(x)

            return parameters, collections, y

        assert total_called == 0

        y, parameters, collections = m.init()(1)

        assert total_called == 1

        parameters, collections, y = f(parameters, collections, 1)

        assert total_called == 2

        parameters, collections, y = f(parameters, collections, 1)

        assert total_called == 2

    def test_jit(self):
        total_called = 0

        class SomeModule(elegy.Module):
            n: jnp.ndarray

            def call(self, x):
                nonlocal total_called

                total_called += 1

                n = self.add_parameter("n", lambda: jnp.array(0))
                self.update_parameter("n", n + 1)

                if elegy.hooks.losses_active():
                    return x + 1
                else:
                    return x - 1

        m = SomeModule()
        assert total_called == 0

        with elegy.hooks.context(losses=True):
            m.init_jit(set_defaults=True)(0)
            # triggers call because its the first time
            assert total_called == 1
            assert m.n == 0

        with elegy.hooks.context(losses=True):
            y = m.call_with_defaults_jit()(0)

            assert y == 1
            assert m.n == 1
            assert total_called == 2

        with elegy.hooks.context(losses=True):
            y = m.call_with_defaults_jit()(0)
            assert m.n == 2
            assert total_called == 2

        with elegy.hooks.context(losses=False):
            y = m.call_with_defaults_jit()(0)
            assert y == -1
            # triggers call because training changed and is static
            assert total_called == 3
            assert m.n == 3

        with elegy.hooks.context(losses=False):
            y = m.call_with_defaults_jit()(0)
            assert y == -1
            # does not trigger call function for training = True exists
            assert total_called == 3
            assert m.n == 4

        with elegy.hooks.context(losses=True):
            y = m.call_with_defaults_jit()(0)
            assert y == 1
            # does not trigger call function for training = True exists
            assert total_called == 3
            assert m.n == 5

        with elegy.hooks.context(losses=True, summaries=True):
            y = m.call_with_defaults_jit()(0)
            assert y == 1
            # triggers call because summaries are now present
            assert total_called == 4
            assert m.n == 6

        with elegy.hooks.context(losses=False, summaries=True):
            y = m.call_with_defaults_jit()(0)
            assert y == -1
            # triggers call because configuration training=False,  is new
            assert total_called == 5
            assert m.n == 7

    def test_jit_auto_init(self):
        total_called = 0

        class SomeModule(elegy.Module):
            n: jnp.ndarray

            def call(self, x):
                nonlocal total_called

                total_called += 1

                n = self.add_parameter("n", lambda: jnp.array(0))
                self.update_parameter("n", n + 1)

                if elegy.hooks.losses_active():
                    return x + 1
                else:
                    return x - 1

        m = SomeModule()

        assert total_called == 0

        with elegy.hooks.context(losses=True):
            m.call_with_defaults_jit()(0)
            assert total_called == 2

            assert m.n == 1

            y = m.call_with_defaults_jit()(0)

            assert y == 1
            assert m.n == 2
            assert total_called == 2

            y = m.call_with_defaults_jit()(0)
            assert m.n == 3
            assert total_called == 2


class TestOthers(TestCase):
    def test_module_system_docs(self):
        class Linear(elegy.Module):
            def __init__(self, n_out):
                super().__init__()
                self.n_out = n_out

            def call(self, x):
                w = self.add_parameter(
                    "w",
                    lambda: elegy.initializers.RandomUniform()(
                        shape=[x.shape[-1], self.n_out],
                        dtype=jnp.float32,
                    ),
                )
                b = self.add_parameter("b", lambda: jnp.zeros(shape=[self.n_out]))

                return jnp.dot(x, w) + b

        class MLP(elegy.Module):
            def call(self, x):
                x = Linear(64)(x)
                x = jax.nn.relu(x)
                x = Linear(32)(x)
                x = jax.nn.relu(x)
                x = Linear(1)(x)
                return x

        def loss_fn(parameters, collections, x, y):
            y_pred, _, collections = mlp.apply(parameters, collections)(x)
            return jnp.mean(jnp.square(y - y_pred))

        def update(parameters, collections, x, y):
            loss, gradients = jax.value_and_grad(loss_fn)(parameters, collections, x, y)
            parameters = jax.tree_multimap(
                lambda p, g: p - 0.01 * g, parameters, gradients
            )

            return parameters

        x = np.random.uniform(size=(15, 3))
        y = np.random.uniform(size=(15, 1))
        mlp = MLP()

        y_pred, parameters, collections = mlp.init(rng=elegy.RNGSeq(42))(x)

        update_jit = jax.jit(update)

        for step in range(1):
            parameters = update_jit(parameters, collections, x, y)

        mlp.set_default_parameters(parameters, collections)
