import jax, jax.numpy as jnp
import numpy as np

import elegy, elegy.slicing
from unittest import TestCase


#TODO: nested modules
#TODO: ensure only necessary modules are executed and only once


class BasicModuleSlicingTest(TestCase):
    def setUp(self):
        self.x = np.random.random((32,100)).astype('float32')
        self.module = BasicModule0()
        self.module.init(rng=elegy.RNGSeq(0), set_defaults=True)(self.x)
        self.model  = elegy.Model(self.module)

    def test_basic_slice_by_name0(self):
        start, end = ("linear0", "linear1")
        submodule = elegy.slicing.slice_model(self.model, start, end, self.x)
        submodel = elegy.Model(submodule)
        assert submodel.predict(self.x).shape == (32, 10)
        assert jnp.all(submodel.predict(self.x) == self.module.test_call0(self.x))

    def test_basic_slice_by_name1(self):
        start, end = (None, "linear1") #None means input
        submodule = elegy.slicing.slice_model(self.model, start, end, self.x)
        submodel = elegy.Model(submodule)
        assert submodel.predict(self.x).shape == (32, 10)
        assert jnp.allclose(submodel.predict(self.x), self.module.test_call1(self.x))
    
    def test_slice_multi_output(self):
        start, end = None, ["linear1", "linear2"]
        submodule = elegy.slicing.slice_model(self.model, start, end, self.x)
        submodel = elegy.Model(submodule)
        
        outputs = submodel.predict(self.x)
        true_outputs = self.module.test_call2(self.x)
        assert len(outputs) == 2
        assert outputs[0].shape == true_outputs[0].shape
        assert outputs[1].shape == true_outputs[1].shape
        assert jnp.allclose(outputs[0], true_outputs[0])
        assert jnp.allclose(outputs[1], true_outputs[1])
    
    def test_slice_return_input(self):
        submodule = elegy.slicing.slice_model(self.model, "input", ["/linear1", "input"], self.x)
        submodel = elegy.Model(submodule)
        submodel.summary(self.x)
        ypred = submodel.predict(self.x)
        assert jnp.all(ypred[1] == self.x)
        assert ypred[0].shape == (32, 10)
        assert jnp.allclose(ypred[0], self.module.test_call1(self.x))
    
    def test_no_path(self):
        for start_module in ["linear2", "linear1"]:
            try:
                submodule = elegy.slicing.slice_model(self.model, start_module, "linear0", self.x)
                submodel = elegy.Model(submodule)
                submodel.summary(self.x)
            except RuntimeError as e:
                assert e.args[0].startswith(f"No path from {start_module} to linear0")
            else:
                assert False, "No error or wrong error raised"


class ResNetSlicingTest(TestCase):
    def test_multi_out(self):
        x = jnp.zeros((2, 224, 224, 3))
        resnet = elegy.nets.resnet.ResNet18()
        resnet.init(rng=elegy.RNGSeq(0), set_defaults=True)(x)
        resnetmodel = elegy.Model(resnet, run_eagerly=True)

        submodule = elegy.slicing.slice_model(
            resnetmodel,
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



class BasicModule0(elegy.Module):
    def call(self, x):
        x = x/255.
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
        x = x/255.
        x = self.linear0.call_with_defaults()(x)
        x = jax.nn.relu(x)
        x = self.linear1.call_with_defaults()(x)
        return x

    def test_call2(self, x):
        x = x/255.
        x = self.linear0.call_with_defaults()(x)
        x = jax.nn.relu(x)
        x = x0 = self.linear1.call_with_defaults()(x)
        x = jax.nn.relu(x)
        x = x1 = self.linear2.call_with_defaults()(x)
        return x0, x1
