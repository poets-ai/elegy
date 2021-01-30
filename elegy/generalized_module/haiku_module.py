import functools
import typing as tp

from haiku._src.base import current_bundle_name

from elegy import hooks, types, utils

from .generalized_module import GeneralizedModule, register_module_for
import toolz

try:
    import haiku
except ImportError:
    raise types.DependencyUnavailable("Flax is not available")


@register_module_for(haiku.Module)
class DummyHaikuModule(GeneralizedModule):
    pass


class HaikuModule(GeneralizedModule):
    def __init__(self, f: tp.Callable):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            kwargs = {
                name[2:] if name.startswith("__") else name: value
                for name, value in kwargs.items()
            }
            return f(*args, **kwargs)

        self.f = wrapper
        self.module = haiku.transform_with_state(wrapper)

    def init(self, rng: types.RNGSeq) -> tp.Callable[..., types.OutputStates]:
        def _lambda(*args, **kwargs):
            def init_fn(*args, **kwargs) -> types.OutputStates:
                kwargs = {f"__{name}": value for name, value in kwargs.items()}
                key = rng.next()
                params, states = self.module.init(key, *args, **kwargs)
                y_pred, _ = self.module.apply(params, states, key, *args, **kwargs)
                return types.OutputStates(y_pred, params, states)

            y_pred, params, states = utils.inject_dependencies(
                init_fn,
                signature_f=self.f,
            )(
                *args,
                **kwargs,
            )

            return types.OutputStates(y_pred, params, states)

        return _lambda

    def apply(
        self,
        params: tp.Any,
        states: tp.Any,
        training: bool,
        rng: types.RNGSeq,
    ) -> tp.Callable[..., types.OutputStates]:
        if params is None:
            params = {}

        if states is None:
            states = {}

        def _lambda(*args, **kwargs):
            def apply_fn(*args, **kwargs):
                kwargs = {f"__{name}": value for name, value in kwargs.items()}
                return self.module.apply(params, states, rng.next(), *args, **kwargs)

            y_pred, states_ = utils.inject_dependencies(apply_fn, signature_f=self.f,)(
                *args,
                **kwargs,
            )

            return types.OutputStates(y_pred, params, states_)

        return _lambda

    def get_summary_params(
        self,
        path: tp.Tuple[str, ...],
        module: tp.Any,
        value: tp.Any,
        include_submodules: bool,
        net_params: types.NetParams,
        net_states: types.NetStates,
    ) -> tp.Tuple[tp.Optional[types.Pytree], tp.Optional[types.Pytree]]:

        path_str = "/".join(path)

        if include_submodules:
            params_tree = {
                name: value for name, value in net_params.items() if path_str in name
            }
            states_tree = {
                name: value for name, value in net_states.items() if path_str in name
            }
        else:
            params_tree = net_params[path_str] if path_str in net_params else None
            states_tree = net_states[path_str] if path_str in net_states else None

        return params_tree, states_tree


def haiku_summarize(f):
    @functools.wraps(f)
    def wrapper(self: haiku.Module, *args, **kwargs):

        outputs = f(self, *args, **kwargs)

        if hooks.summaries_active():
            path = current_bundle_name().split("/")
            hooks.add_summary(tuple(path), self, outputs)

        return outputs

    return wrapper


def haiku_summary(
    name: str,
    f: tp.Any,
    value: types.Scalar,
):
    if hooks.summaries_active():
        path = tuple(current_bundle_name().split("/")) + (name,)
        hooks.add_summary(path, f, value)


def assert_id(value):
    assert value
    return value
