import typing as tp

from elegy import utils
from elegy.types import OutputStates, RNGSeq
from elegy import module, hooks

from .generalized_module import GeneralizedModule, register_module_for


@register_module_for(module.Module)
class ElegyModule(GeneralizedModule):
    def __init__(self, module: module.Module):
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
