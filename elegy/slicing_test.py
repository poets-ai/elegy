import jax, jax.numpy as jnp
import elegy, elegy.slicing
from unittest import TestCase





class ModuleSlicingTest(TestCase):
    def test_basic_slice_by_name0(self):
        x = jnp.zeros((32, 100))

        start, end = ("linear0", "linear1")
        basicmodule = BasicModule0()
        basicmodel  = elegy.Model(basicmodule)
        submodule = elegy.slicing.slice_model(basicmodel, start, end, x)
        submodel = elegy.Model(submodule)
        basicmodule.init(rng=elegy.RNGSeq(0), set_defaults=True)(x)
        assert submodel.predict(x).shape == (32, 10)
        assert jnp.all(submodel.predict(x) == basicmodule.test_call0(x))

    
    def test_basic_slice_by_name1(self):
        x = jnp.zeros((32, 100))

        start, end = (None, "linear1") #None means input
        basicmodule = BasicModule0()
        basicmodel  = elegy.Model(basicmodule)
        submodule = elegy.slicing.slice_model(basicmodel, start, end, x)
        submodel = elegy.Model(submodule)
        basicmodule.init(rng=elegy.RNGSeq(0), set_defaults=True)(x)
        # submodel.summary(x)
        assert submodel.predict(x).shape == (32, 10)
        assert jnp.all(submodel.predict(x) == basicmodule.test_call1(x))
    
    






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
