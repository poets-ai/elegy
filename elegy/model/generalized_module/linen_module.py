import typing as tp


from elegy import utils
from elegy.types import (
    DependencyUnavailable,
    NetParams,
    NetStates,
    OutputStates,
    Path,
    Pytree,
    RNGSeq,
)

from .generalized_module import GeneralizedModule, register_module_for

try:
    import flax.linen as nn
    from flax import linen
    from flax.core import FrozenDict, freeze, unfreeze
except ImportError:
    raise DependencyUnavailable("Flax is not available")


@register_module_for(nn.Module)
class LinenModule(GeneralizedModule):
    def __init__(self, module: nn.Module):
        self.module = module

    def init(self, rng: RNGSeq) -> tp.Callable[..., OutputStates]:
        def _lambda(*args, **kwargs):
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

            net_states, net_params = (
                variables.pop("params")
                if "params" in variables
                else (variables, FrozenDict())
            )

            return OutputStates(y_pred, net_params, net_states)

        return _lambda

    def apply(
        self,
        params: tp.Any,
        states: tp.Any,
        rng: RNGSeq,
    ) -> tp.Callable[..., OutputStates]:
        if params is None:
            params = FrozenDict()

        if states is None:
            states = FrozenDict()

        def _lambda(*args, **kwargs):
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

            net_states, net_params = (
                variables.pop("params")
                if "params" in variables
                else (variables, FrozenDict())
            )

            return OutputStates(y_pred, net_params, net_states)

        return _lambda

    def get_summary_params(
        self,
        path: Path,
        net_params: NetParams,
        net_states: NetStates,
    ) -> tp.Tuple[Pytree, Pytree]:
        return None, None
