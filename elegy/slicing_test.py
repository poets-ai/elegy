import jax, jax.numpy as jnp
import numpy as np

import elegy, elegy.slicing
import optax
from unittest import TestCase


class BasicModuleSlicingTest(TestCase):
    def setUp(self):
        self.x = np.random.random((32, 100)).astype("float32")
        self.module = BasicModule0()
        self.module.init(rng=elegy.RNGSeq(0), set_defaults=True)(self.x)

    def test_basic_slice_by_name0(self):
        start, end = ("linear0", "linear1")
        submodule = self.module.slice(start, end, self.x)
        submodel = elegy.Model(submodule)
        assert submodel.predict(self.x, initialize=True).shape == (32, 10)
        assert jnp.all(submodel.predict(self.x) == self.module.test_call0(self.x))
        # different batch size
        assert submodel.predict(self.x[:8]).shape == (8, 10)

    def test_basic_slice_by_name1(self):
        start, end = (None, "linear1")  # None means input
        submodule = self.module.slice(start, end, self.x)
        submodel = elegy.Model(submodule)
        assert submodel.predict(self.x, initialize=True).shape == (32, 10)
        assert jnp.allclose(submodel.predict(self.x), self.module.test_call1(self.x))

    def test_slice_multi_output(self):
        start, end = None, ["linear2", "linear1"]
        submodule = self.module.slice(start, end, self.x)
        submodel = elegy.Model(submodule)

        outputs = submodel.predict(self.x, initialize=True)
        true_outputs = self.module.test_call2(self.x)
        assert len(outputs) == 2
        assert outputs[0].shape == true_outputs[0].shape
        assert outputs[1].shape == true_outputs[1].shape
        assert jnp.allclose(outputs[0], true_outputs[0])
        assert jnp.allclose(outputs[1], true_outputs[1])

    def test_slice_return_input(self):
        submodule = self.module.slice("input", ["/linear1", "input"], self.x)
        submodel = elegy.Model(submodule)
        submodel.summary(self.x)
        ypred = submodel.predict(self.x, initialize=True)
        assert jnp.all(ypred[1] == self.x)
        assert ypred[0].shape == (32, 10)
        assert jnp.allclose(ypred[0], self.module.test_call1(self.x))
        assert "linear2" not in submodel.states["net_params"].keys()

    def test_no_path(self):
        for start_module in ["linear2", "linear1"]:
            try:
                submodule = self.module.slice(start_module, "linear0", self.x)
                submodel = elegy.Model(submodule)
                submodel.summary(self.x)
            except RuntimeError as e:
                assert e.args[0].startswith(f"No path from {start_module} to linear0")
            else:
                assert False, "No error or wrong error raised"

    def test_retrain(self):
        y = jnp.zeros((32, 10))

        submodule = self.module.slice("linear0", "linear1", self.x)
        submodel = elegy.Model(
            submodule,
            loss=elegy.losses.MeanAbsoluteError(),
            optimizer=optax.sgd(0.05),
        )
        submodel.init(self.x, y)
        y0 = submodel.predict(self.x)

        submodel.fit(self.x, y, epochs=3, verbose=2)

        y2 = submodel.predict(self.x)
        # output after training should be closer to zero because targets are zero
        assert jnp.abs(y2.mean()) < jnp.abs(y0.mean())


class ResNetSlicingTest(TestCase):
    def test_multi_out(self):
        x = jnp.zeros((2, 224, 224, 3))
        resnet = elegy.nets.resnet.ResNet18()
        resnet.init(rng=elegy.RNGSeq(0), set_defaults=True)(x)

        submodule = resnet.slice(
            start=None,
            end=[
                "/res_net_block_1",
                "/res_net_block_3",
                "/res_net_block_5",
                "/res_net_block_6",
                "/res_net_block_7",
            ],
            sample_input=x,
        )
        submodel = elegy.Model(submodule, run_eagerly=True)

        # submodel.summary(x)
        outputs = submodel.predict(x, initialize=True)
        print(jax.tree_map(jnp.shape, outputs))
        assert len(outputs) == 5
        assert outputs[0].shape == (2, 56, 56, 64)
        assert outputs[1].shape == (2, 28, 28, 128)
        assert outputs[2].shape == (2, 14, 14, 256)
        assert outputs[3].shape == (2, 7, 7, 512)
        assert outputs[4].shape == (2, 7, 7, 512)


class NestedSlicingTest(TestCase):
    def test_basic_nested(self):
        self.x = np.random.random((32, 100)).astype("float32")
        self.module = NestedModule0()
        self.module.init(rng=elegy.RNGSeq(0), set_defaults=True)(self.x)

        # self.model.summary(self.x)
        submodule = self.module.slice("/module0/linear1", "/module1/linear1", self.x)
        submodel = elegy.Model(submodule)

        x_for_submodel = np.random.random([16, 25])
        submodel.predict(x_for_submodel, initialize=True)
        submodel.summary(x_for_submodel)

        assert jnp.allclose(
            submodel.predict(x_for_submodel), self.module.test_call0(x_for_submodel)
        )
        assert "module0_linear1" in submodel.states["net_params"].keys()
        assert "module0_linear0" not in submodel.states["net_params"].keys()
        assert "module1_linear2" not in submodel.states["net_params"].keys()


def test_no_default_parameters():
    x = np.random.random((32, 100)).astype("float32")
    module = BasicModule0()
    model = elegy.Model(module, seed=np.random.randint(100, 100000))
    model.init(x)
    model.update_modules()

    submodel = elegy.Model(model.module.slice("linear0", "linear1", x))
    assert submodel.predict(x, initialize=True).shape == (32, 10)

    assert jnp.allclose(submodel.predict(x), module.test_call0(x))


class BasicModule0(elegy.Module):
    def call(self, x):
        x = x / 255.0
        x = elegy.nn.Linear(25, name="linear0")(x)
        x = jax.nn.relu(x)
        x = elegy.nn.Linear(10, name="linear1")(x)
        x = jax.nn.relu(x)
        x = elegy.nn.Linear(5, name="linear2")(x)
        return x

    def test_call0(self, x):
        x = self.linear0.call_with_defaults()(x)
        x = jax.nn.relu(x)
        x = self.linear1.call_with_defaults()(x)
        return x

    def test_call1(self, x):
        x = x / 255.0
        x = self.linear0.call_with_defaults()(x)
        x = jax.nn.relu(x)
        x = self.linear1.call_with_defaults()(x)
        return x

    def test_call2(self, x):
        x = x / 255.0
        x = self.linear0.call_with_defaults()(x)
        x = jax.nn.relu(x)
        x = x0 = self.linear1.call_with_defaults()(x)
        x = jax.nn.relu(x)
        x = x1 = self.linear2.call_with_defaults()(x)
        return x1, x0


class NestedModule0(elegy.Module):
    def call(self, x):
        x = BasicModule0(name="module0")(x)
        x = x * 255
        x = BasicModule0(name="module1")(x)
        return x

    def test_call0(self, x):
        x = self.module0.linear1.call_with_defaults()(x)
        x = jax.nn.relu(x)
        x = self.module0.linear2.call_with_defaults()(x)
        x = x * 255
        x = x / 255.0
        x = self.module1.linear0.call_with_defaults()(x)
        x = jax.nn.relu(x)
        x = self.module1.linear1.call_with_defaults()(x)
        return x
