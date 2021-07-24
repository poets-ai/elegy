import functools
import typing as tp

import jax
from elegy import hooks, types, utils
from rich.table import Table
from rich.text import Text
from toolz.functoolz import apply

from .generalized_module import GeneralizedModule, register_module_for

try:
    import flax.linen as nn
    from flax import linen
    from flax.core import FrozenDict, freeze, unfreeze
except ImportError:
    raise types.DependencyUnavailable("Flax is not available")


@register_module_for(nn.Module)
class LinenModule(GeneralizedModule):
    def __init__(self, module: nn.Module):
        self.module = module

    def init(self, rng: types.RNGSeq) -> tp.Callable[..., types.OutputStates]:
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

            return types.OutputStates(y_pred, net_params, net_states)

        return _lambda

    def apply(
        self,
        params: tp.Any,
        states: tp.Any,
        training: bool,
        rng: types.RNGSeq,
    ) -> tp.Callable[..., types.OutputStates]:
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

            return types.OutputStates(y_pred, net_params, net_states)

        return _lambda

    def summary(
        self,
        x: tp.Any,
        x_args: tp.Tuple,
        x_kwargs: tp.Dict[str, tp.Any],
        params: tp.Any,
        states: tp.Any,
        rng: types.RNGSeq,
        depth: int,
        run_eagerly: bool,
        eval_shape: bool,
    ) -> str:

        apply_fn = self.apply(
            params=params,
            states=states,
            training=False,
            rng=rng,
        )

        if eval_shape:
            ouput_states: types.OutputStates = jax.eval_shape(
                apply_fn,
                *x_args,
                **x_kwargs,
            )
        else:
            ouput_states = apply_fn(
                *x_args,
                **x_kwargs,
            )

        # summary string
        summary = "\n"

        # input / output shapes table
        values_table = Table(
            title="",
            show_header=False,
            show_lines=True,
            # show_footer=True,
            # box=rich.box.HORIZONTALS,
        )

        values_table.add_column("Layer")
        values_table.add_column("Shape")

        values_rows = [
            ["Inputs", utils.format_output(x)],
            ["Outputs", utils.format_output(ouput_states.preds)],
        ]

        utils.add_padding(values_rows)

        for row in values_rows:
            values_table.add_row(*row)

        summary += utils.get_table_repr(values_table) + "\n"

        # params table
        params_table = Table(
            title="Parameters",
            show_header=True,
            show_lines=True,
            # show_footer=True,
            # box=rich.box.HORIZONTALS,
        )