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
        submodule = basicmodule.slice(basicmodule.linear0, basicmodule.linear1, x)
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
            submodule = basicmodule.slice(start, end, x)
            submodel = elegy.Model(submodule)
            submodel.summary(x)
            assert submodel.predict(x).shape == (32, 10)
            assert jnp.all(submodel.predict(x) == basicmodule.test_call(x))

    def test_slice_return_input(self):
        x = jnp.zeros((32, 100))
        basicmodule = BasicModule0()
        submodule = basicmodule.slice("input", ["/linear1", "input"], x)
        submodel = elegy.Model(submodule)
        submodel.summary(x)
        ypred = submodel.predict(x)
        assert jnp.all(ypred[1] == x)
        assert ypred[0].shape == (32, 10)
        assert jnp.all(ypred[0] == basicmodule.test_call(x))

    def test_resnet_multi_out(self):
        x = jnp.zeros((2, 224, 224, 3))
        resnet = elegy.nets.resnet.ResNet18()
        submodule = resnet.slice(
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
        submodule = basicmodule.slice("linear0", "linear1", x)
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
        for start_module in ["linear2", "linear1"]:
            try:
                submodule = basicmodule.slice(start_module, "linear0", x)
            except RuntimeError as e:
                assert e.args[0].startswith(f"No path from /{start_module} to /linear0")
            else:
                assert False, "No error or wrong error raised"

    def test_multi_input_modules(self):
        x = jnp.ones((32, 100))

        module = ContainsMultiInputModule()
        model = elegy.Model(module)
        model.summary(x)

        submodule = module.slice(None, "/multi_input_module", x)
        submodel = elegy.Model(submodule)
        submodel.summary(x)
        print(submodule.get_parameters())

        y = submodel.predict(x)
        print(y.shape)
        assert y.shape == (32, 25)
        assert jnp.allclose(y, module.test_call(x))

    def test_computationally_equivalent_paths(self):
        import networkx as nx

        G = nx.DiGraph()
        G.add_edge(0, 1, inkey=0)
        G.add_edge(1, 2, inkey=0)
        G.add_edge(0, 2, inkey=0)  # 0->2 is equivalent to the path 0->1->2
        G.add_edge(2, 3, inkey=0)
        G.add_edge(3, 4, inkey=0)

        g0 = G.edge_subgraph([(0, 1), (1, 2), (2, 3)]).copy()
        g1 = G.edge_subgraph([(0, 2), (2, 3)]).copy()

        apce = elegy.module_slicing.are_paths_computationally_equivalent
        fcep = elegy.module_slicing.filter_computationally_equivalent_paths

        assert apce(g0, g1)
        assert apce(g1, g0)
        filtered_paths = fcep([g0, g1])
        assert len(filtered_paths) == 1
        assert filtered_paths[0] == g1

        G = nx.DiGraph()
        G.add_edge(0, 1, inkey=0)
        G.add_edge(1, 2, inkey=0)
        G.add_edge(0, 2, inkey=1)  # not equivalent, multi-input module
        G.add_edge(2, 3, inkey=0)
        G.add_edge(3, 4, inkey=0)

        g0 = G.edge_subgraph([(0, 1), (1, 2), (2, 3)]).copy()
        g1 = G.edge_subgraph([(0, 2), (2, 3)]).copy()
        g2 = G.edge_subgraph([(0, 2), (2, 3), (3, 4)]).copy()

        apce = elegy.module_slicing.are_paths_computationally_equivalent
        assert not apce(g0, g1)
        assert not apce(g1, g0)
        assert not apce(g1, g2)
        filtered_paths = fcep([g0, g1, g2])
        assert len(filtered_paths) == 3
        assert g0 in filtered_paths and g1 in filtered_paths and g2 in filtered_paths

    def test_split_merge_args_kwargs(self):
        args_kwargs = elegy.module_slicing.merge_args_kwargs(0, 101, -2, a=65, b=77)
        assert len(args_kwargs) == 5
        for x in [(0, 0), (1, 101), (2, -2), ("a", 65), ("b", 77)]:
            assert x in args_kwargs

        args, kwargs = elegy.module_slicing.split_merged_args_kwargs(args_kwargs)
        assert args == (0, 101, -2)
        assert len(kwargs) == 2
        assert kwargs["a"] == 65 and kwargs["b"] == 77


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


class MultiInputModule(elegy.Module):
    def call(self, x0, x1):
        return x0[..., :25] + x1[..., :25]


class ContainsMultiInputModule(elegy.Module):
    def call(self, x):
        x0 = elegy.nn.Linear(25, name="linear0")(x)
        x = MultiInputModule(name="multi_input_module")(x, x0)
        x = elegy.nn.Linear(10)(x)
        return x

    def test_call(self, x):
        x0 = self.linear0(x)
        x = self.multi_input_module(x, x0)
        return x
