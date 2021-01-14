import typing as tp

import flax.linen as nn
from elegy import utils
from elegy.types import OutputStates
from flax import linen
from flax.core import FrozenDict, freeze, unfreeze

from .generalized_module import GeneralizedModule, register_module


class LinenModule(GeneralizedModule):
    def __init__(self, module: nn.Module):
        self.module = module

    @classmethod
    def new(cls, module: nn.Module) -> GeneralizedModule:
        return cls(module)

    def init(
        self, rng: utils.RNGSeq, args: tp.Tuple, kwargs: tp.Dict[str, tp.Any]
    ) -> OutputStates:
        def init_fn(*args, **kwargs):
            return self.module.init_with_output(rng.next(), *args, **kwargs)

        y_pred, variables = utils.inject_dependencies(
            init_fn,
            signature_f=self.module.__call__,
        )(
            *args,
            **kwargs,
        )
        assert isinstance(variables, FrozenDict)

        net_states, net_params = variables.pop("params")

        return OutputStates(y_pred, net_params, net_states)

    def apply(
        self,
        params: tp.Any,
        states: tp.Any,
        rng: utils.RNGSeq,
        args: tp.Tuple,
        kwargs: tp.Dict[str, tp.Any],
    ) -> OutputStates:
        def apply_fn(*args, **kwargs):
            variables = dict(params=params, **states)
            return self.module.apply(
                variables,
                *args,
                rngs={"params": rng.next()},
                mutable=True,
                **kwargs,
            )

        y_pred, variables = utils.inject_dependencies(
            apply_fn,
            signature_f=self.module.__call__,
        )(
            *args,
            **kwargs,
        )

        net_states, net_params = variables.pop("params")

        return OutputStates(y_pred, net_params, net_states)


register_module(nn.Module, LinenModule)
