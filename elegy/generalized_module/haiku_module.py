import functools
import typing as tp

import jax
import jax.numpy as jnp
import toolz
from elegy import hooks, types, utils
from rich.table import Table
from rich.text import Text

from .generalized_module import GeneralizedModule, register_module_for

try:
    import haiku
    from haiku._src.base import current_bundle_name, new_context
    from haiku._src.transform import (
        APPLY_RNG_ERROR,
        APPLY_RNG_STATE_ERROR,
        INIT_RNG_ERROR,
        check_mapping,
        to_prng_sequence,
    )
except ImportError:
    raise types.DependencyUnavailable("'haiku' is not available")


@register_module_for(haiku.Module)
class DummyHaikuModule(GeneralizedModule):
    pass


class TransformWithStateAndOutput(tp.NamedTuple):
    f: tp.Callable

    def init(
        self,
        rng: tp.Optional[tp.Union[jnp.ndarray, int]],
        *args,
        **kwargs,
    ) -> tp.Tuple[tp.Any, haiku.Params, haiku.State]:
        """Initializes your function collecting parameters and state."""
        rng = to_prng_sequence(rng, err_msg=INIT_RNG_ERROR)
        with new_context(rng=rng) as ctx:
            output = self.f(*args, **kwargs)
        return output, ctx.collect_params(), ctx.collect_initial_state()

    def apply(
        self,
        params: tp.Optional[haiku.Params],
        state: tp.Optional[haiku.State],
        rng: tp.Optional[tp.Union[jnp.ndarray, int]],
        *args,
        **kwargs,
    ) -> tp.Tuple[tp.Any, haiku.State]:
        """Applies your function injecting parameters and state."""
        params = check_mapping("params", params)
        state = check_mapping("state", state)
        rng = to_prng_sequence(
            rng, err_msg=(APPLY_RNG_STATE_ERROR if state else APPLY_RNG_ERROR)
        )
        with new_context(params=params, state=state, rng=rng) as ctx:
            out = self.f(*args, **kwargs)
        return out, ctx.collect_state()


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
        self.module = TransformWithStateAndOutput(wrapper)

    def init(self, rng: types.RNGSeq) -> tp.Callable[..., types.OutputStates]:
        def _lambda(*args, **kwargs):
            def init_fn(*args, **kwargs) -> types.OutputStates:
                kwargs = {f"__{name}": value for name, value in kwargs.items()}
                key = rng.next()
                y_pred, params, states = self.module.init(key, *args, **kwargs)
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

    def summary(
        self,
        x: tp.Any,
        depth: int,
        run_eagerly: bool,
        eval_shape: bool,
    ) -> str:
        def summary_fn(*args, **kwargs):
            kwargs = {f"__{name}": value for name, value in kwargs.items()}
            return self.f(*args, **kwargs)

        x_args, x_kwargs = utils.get_input_args(
            x,
            states=types.States(rng=types.RNGSeq(42)),
            initializing=True,
            training=True,
        )

        summary = utils.inject_dependencies(
            haiku.experimental.tabulate(summary_fn),
            signature_f=self.f,
        )(*x_args, **x_kwargs)

        return summary
