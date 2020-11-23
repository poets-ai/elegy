import elegy
import elegy.module_slicing
from unittest import TestCase
import jax, jax.numpy as jnp


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
        # assert False


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
