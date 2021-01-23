import typing as tp

from elegy import utils
from elegy.types import (
    NetParams,
    NetStates,
    OutputStates,
    Path,
    Pytree,
    RNGSeq,
    SummaryModule,
    SummaryValue,
)
from elegy.module import Module
from elegy import hooks

from .generalized_module import GeneralizedModule, register_module_for


@register_module_for(Module)
class ElegyModule(GeneralizedModule):
    def __init__(self, module: Module):
        self.module = module

    def init(self, rng: RNGSeq) -> tp.Callable[..., OutputStates]:
        def _lambda(*args, **kwargs):

            y_pred, collections = utils.inject_dependencies(self.module.init(rng=rng))(
                *args,
                **kwargs,
            )
            assert isinstance(collections, dict)

            net_params = collections.pop("parameters", {})
            net_states = collections

            return OutputStates(y_pred, net_params, net_states)

        return _lambda

    def apply(
        self,
        params: tp.Any,
        states: tp.Any,
        rng: RNGSeq,
    ) -> tp.Callable[..., OutputStates]:
        def _lambda(*args, **kwargs):
            collections = states.copy() if states is not None else {}
            if params is not None:
                collections["parameters"] = params

            y_pred, collections = utils.inject_dependencies(
                self.module.apply(collections, rng=rng),
            )(
                *args,
                **kwargs,
            )
            assert isinstance(collections, dict)

            net_params = collections.pop("parameters", {})
            net_states = collections

            return OutputStates(y_pred, net_params, net_states)

        return _lambda

    def get_summary_params(
        self,
        path: Path,
        module: tp.Any,
        value: tp.Any,
        include_submodules: bool,
        net_params: NetParams,
        net_states: NetStates,
    ) -> tp.Tuple[tp.Optional[Pytree], tp.Optional[Pytree]]:

        if net_params is None:
            params_tree = None
        else:
            params_tree = utils.get_path_params(path, net_params)
            # filter only params
            if not include_submodules and params_tree is not None:
                params_tree = {
                    name: value
                    for name, value in params_tree.items()
                    if name in module._params
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
                        if name in module._params
                    }
                    for collection, states in states_tree.items()
                    if states is not None
                }

        return params_tree, states_tree
