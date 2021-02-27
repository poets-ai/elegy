import jax, jax.numpy as jnp
import numpy as np

import elegy, elegy.slicing
from unittest import TestCase





class ModuleSlicingTest(TestCase):
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
