import typing as tp

from elegy import hooks, types, utils
from elegy.module import Module

from .generalized_module import GeneralizedModule, register_module_for


@register_module_for(Module)
class ElegyModule(GeneralizedModule):
    def __init__(self, module: Module):
        self.module = module

    def init(self, rng: types.RNGSeq) -> tp.Callable[..., types.OutputStates]:
        def _lambda(*args, **kwargs):

            y_pred, collections = utils.inject_dependencies(self.module.init(rng=rng))(
                *args,
                **kwargs,
            )
            assert isinstance(collections, dict)

            net_params = collections.pop("parameters", {})
            net_states = collections

            return types.OutputStates(y_pred, net_params, net_states)

        return _lambda

    def update(
        self,
        params: tp.Optional[types.ModuleParams],
        states: tp.Optional[types.ModuleStates],
    ) -> tp.Tuple[tp.Optional[types.ModuleParams], tp.Optional[types.ModuleStates]]:

        collections = states

        if not utils.none_or_uninitialized(params):
            collections = collections.copy() if collections is not None else {}
            collections["parameters"] = params

        if not utils.none_or_uninitialized(collections):
            assert collections is not None
            self.module.set_default_parameters(collections)

        return params, states

    def apply(
        self,
        params: tp.Any,
        states: tp.Any,
        training: bool,
        rng: types.RNGSeq,
    ) -> tp.Callable[..., types.OutputStates]:
        def _lambda(*args, **kwargs):
            collections = states.copy() if states is not None else {}
            if params is not None:
                collections["parameters"] = params

            y_pred, collections = utils.inject_dependencies(
                self.module.apply(collections, training=training, rng=rng),
            )(
                *args,
                **kwargs,
            )
            assert isinstance(collections, dict)

            net_params = collections.pop("parameters", {})
            net_states = collections

            return types.OutputStates(y_pred, net_params, net_states)

        return _lambda

    def get_summary_params(
        self,
        path: types.Path,
        module: tp.Any,
        value: tp.Any,
        include_submodules: bool,
        net_params: types.NetParams,
        net_states: types.NetStates,
    ) -> tp.Tuple[tp.Optional[types.Pytree], tp.Optional[types.Pytree]]:

        if net_params is None:
            params_tree = None
        else:
            params_tree = utils.get_path_params(path, net_params)
            # filter only params
            if not include_submodules and params_tree is not None:
                assert isinstance(module, Module)
                params_tree = {
                    name: value
                    for name, value in params_tree.items()
                    if name in module._spec
                }

        if net_states is None:
            states_tree = None
        else:
            states_tree = {
                collection: utils.get_path_params(path, states)
                for collection, states in net_states.items()
            }
            # filter only params
            if not include_submodules:
                states_tree = {
                    collection: {
                        name: value
                        for name, value in states.items()
                        if name in module._spec
                    }
                    for collection, states in states_tree.items()
                    if states is not None
                }

        return params_tree, states_tree
