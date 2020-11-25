import elegy
import elegy.module_slicing
from unittest import TestCase
import jax, jax.numpy as jnp
import optax


class ModuleSlicingTest(TestCase):
    def test_basic_slice_by_ref(self):
        x = jnp.zeros((32, 100))
        basicmodule = BasicModule0()
        basicmodule(x)  # trigger creation of weights and submodules
        submodule = elegy.module_slicing.slice_module_from_to(
            basicmodule, basicmodule.linear0, basicmodule.linear1, x
        )
        submodel = elegy.Model(submodule)
        submodel.summary(x)
        assert submodel.predict(x).shape == (32, 10)
        assert jnp.all(submodel.predict(x) == basicmodule.test_call(x))

    def test_basic_slice_by_name(self):
        x = jnp.zeros((32, 100))
        START_END_COMBOS = [("linear0", "linear1"), (None, "/linear1")]
        for start, end in START_END_COMBOS:
            print(start, end)
            basicmodule = BasicModule0()
            submodule = elegy.module_slicing.slice_module_from_to(
                basicmodule, start, end, x
            )
            submodel = elegy.Model(submodule)
            submodel.summary(x)
            assert submodel.predict(x).shape == (32, 10)
            assert jnp.all(submodel.predict(x) == basicmodule.test_call(x))

    def test_resnet_multi_out(self):
        x = jnp.zeros((2, 224, 224, 3))
        resnet = elegy.nets.resnet.ResNet18()
        submodule = elegy.module_slicing.slice_module_from_to(
            resnet,
            start_module=None,
            end_module=[
                "/res_net_block_1",
                "/res_net_block_3",
                "/res_net_block_5",
                "/res_net_block_6",
                "/res_net_block_7",
            ],
            sample_input=x,
        )
        submodel = elegy.Model(submodule)
        # submodel.summary(x)
        outputs = submodel.predict(x)
        print(jax.tree_map(jnp.shape, outputs))
        assert len(outputs) == 5
        assert outputs[0].shape == (2, 56, 56, 64)
        assert outputs[1].shape == (2, 28, 28, 128)
        assert outputs[2].shape == (2, 14, 14, 256)
        assert outputs[3].shape == (2, 7, 7, 512)
        assert outputs[4].shape == (2, 7, 7, 512)

        print(jax.tree_map(jnp.shape, resnet.get_parameters()))
        print(jax.tree_map(jnp.shape, submodel.get_parameters()))

    def test_retrain(self):
        x = jnp.ones((32, 100))
        y = jnp.zeros((32, 10))

        basicmodule = BasicModule0()
        submodule = elegy.module_slicing.slice_module_from_to(
            basicmodule, "linear0", "linear0", x
        )
        submodel = elegy.Model(
            submodule,
            loss=elegy.losses.MeanAbsoluteError(),
            optimizer=optax.adamw(1e-3),
        )
        y0 = submodel.predict(x)
        y1 = basicmodule.test_call(x)

        submodel.fit(x, y, epochs=3, verbose=2)

        y2 = submodel.predict(x)
        y3 = basicmodule.test_call(x)

        assert jnp.all(y2 == y3)
        # output after training should be closer to zero because targets are zero
        assert jnp.abs(y2.mean()) < jnp.abs(y0.mean())

    def test_no_path(self):
        x = jnp.ones((32, 100))
        basicmodule = BasicModule0()
        try:
            submodule = elegy.module_slicing.slice_module_from_to(
                basicmodule, "linear2", "linear0", x
            )
        except RuntimeError as e:
            assert e.args[0].startswith("No path from /linear2 to /linear0")
        else:
            assert False, "No error or wrong error raised"

        try:
            submodule = elegy.module_slicing.slice_module_from_to(
                basicmodule, "linear1", "linear0", x
            )
        except RuntimeError as e:
            assert e.args[0].startswith(
                "No operations between the input of /linear1 and the output of /linear0"
            )
        else:
            assert False, "No error or wrong error raised"


class BasicModule0(elegy.Module):
    def call(self, x):
        x = elegy.nn.Linear(25, name="linear0")(x)
        x = elegy.nn.Linear(10, name="linear1")(x)
        x = elegy.nn.Linear(5, name="linear2")(x)
        return x

    def test_call(self, x):
        x = self.linear0(x)
        x = self.linear1(x)
        return x
