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

            y_pred, parameters, collections = utils.inject_dependencies(
                self.module.init(rng=rng)
            )(
                *args,
                **kwargs,
            )

            return types.OutputStates(y_pred, parameters, collections)

        return _lambda

        return params, states

    def apply(
        self,
        params: tp.Any,
        states: tp.Any,
        training: bool,
        rng: types.RNGSeq,
    ) -> tp.Callable[..., types.OutputStates]:
        def _lambda(*args, **kwargs):

            y_pred, net_params, net_states = utils.inject_dependencies(
                self.module.apply(params, states, training=training, rng=rng),
            )(
                *args,
                **kwargs,
            )

            return types.OutputStates(y_pred, net_params, net_states)

        return _lambda

    def update(
        self,
        params: tp.Optional[types.ModuleParams],
        states: tp.Optional[types.ModuleStates],
    ):
        if states is not None:
            self.module.set_default_parameters(params, states)

    def get_summary_params(
        self,
        path: types.Path,
        module: tp.Any,
        value: tp.Any,
        net_params: types.NetParams,
        net_states: types.NetStates,
    ) -> tp.Tuple[tp.Optional[types.Pytree], tp.Optional[types.Pytree]]:

        if net_params is None:
            params_tree = None
        else:
            params_tree = utils.get_path_params(path, net_params)
            # filter out submodules
            if params_tree is not None:
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
            # filter out submodules
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
