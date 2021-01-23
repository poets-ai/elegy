import functools
import typing as tp

from elegy import hooks, utils
from elegy.types import (
    DependencyUnavailable,
    NetParams,
    NetStates,
    OutputStates,
    Path,
    Pytree,
    RNGSeq,
    Scalar,
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
                assert isinstance(module, linen.Module)
                params_tree = {
                    name: value
                    for name, value in params_tree.items()
                    if not name[0].isupper()
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
                        if assert_id(isinstance(module, linen.Module))
                        and not name[0].isupper()
                    }
                    for collection, states in states_tree.items()
                    if states is not None
                }

        return params_tree, states_tree


def flax_summarize(f):
    @functools.wraps(f)
    def wrapper(self: linen.Module, *args, **kwargs):

        outputs = f(self, *args, **kwargs)

        if hooks.summaries_active():
            path = self.scope.path
            hooks.add_summary(path, self, outputs)

        return outputs

    return wrapper


def flax_summary(
    flax_module: linen.Module,
    name: str,
    f: tp.Any,
    value: Scalar,
):
    if hooks.summaries_active():
        path = flax_module.scope.path + (name,)
        hooks.add_summary(path, f, value)


def assert_id(value):
    assert value
    return value