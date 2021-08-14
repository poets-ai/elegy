import typing as tp

import jax
import toolz
from elegy import types, utils
from rich.table import Table
from rich.text import Text

from .generalized_module import GeneralizedModule, register_module_for

try:
    from flax import linen
    from flax.core import FrozenDict
except ImportError:
    raise types.DependencyUnavailable("Flax is not available")


@register_module_for(linen.Module)
class LinenModule(GeneralizedModule):
    def __init__(self, module: linen.Module):
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
                    rngs={"params": rng.next(), "dropout": rng.next()},
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
        depth: int,
        run_eagerly: bool,
        eval_shape: bool,
    ) -> str:
        def apply_fn(x) -> types.OutputStates:
            rng = types.RNGSeq(42)
            x_args, x_kwargs = utils.get_input_args(
                x,
                states=types.States(rng=rng),
                initializing=True,
                training=True,
            )
            return self.init(rng=rng)(*x_args, **x_kwargs)

        if eval_shape:
            ouput_states: types.OutputStates = jax.eval_shape(apply_fn, x)
        elif run_eagerly:
            ouput_states = apply_fn(x)
        else:
            # Not sure passing params and states as captures is the best way to do this
            # however, since we are just using this function once it should be fine.
            ouput_states = jax.jit(apply_fn)(x)

        preds, params, states = ouput_states

        # summary string
        summary = "\n"

        # input / output shapes table
        shapes_table = Table(
            title="Shapes",
            show_header=False,
            show_lines=True,
            title_justify="left",
            # show_footer=True,
            # box=rich.box.HORIZONTALS,
        )

        shapes_table.add_column("Layer")
        shapes_table.add_column("Shape")

        shapes_rows = [
            ["Input", utils.format_output(x)],
            ["Output", utils.format_output(preds)],
        ]

        utils.add_padding(shapes_rows)

        for row in shapes_rows:
            shapes_table.add_row(*row)

        summary += utils.get_table_repr(shapes_table)

        # params table
        variables = dict(params=params, **states)
        param_types = list(variables.keys())

        params_table = Table(
            title="Parameters",
            show_header=True,
            show_lines=True,
            title_justify="left",
            show_footer=True,
            # box=rich.box.HORIZONTALS,
        )

        params_table.add_column("path")
        for param_type in param_types:
            params_table.add_column(param_type)

        params_rows = self.params_rows(
            depth=depth, variables=variables, param_types=param_types
        )

        # add totals row
        totals = ["Total"] + [
            utils.format_count_and_size(params) for params in variables.values()
        ]
        params_rows.append(totals)

        utils.add_padding(params_rows)

        for row in params_rows[:-1]:
            params_table.add_row(*row)

        # totals as footer
        for i, val in enumerate(params_rows[-1]):
            params_table.columns[i].footer = (
                Text.from_markup(val, justify="right") if i == 0 else val
            )

        # all params total as caption
        params_table.caption_style = "bold"
        params_table.caption = "\nTotal Parameters: " + utils.format_count_and_size(
            variables, add_padding=False
        )

        summary += "\n" + utils.get_table_repr(params_table)

        return summary

    def params_rows(
        self, depth: int, variables: tp.Dict[str, tp.Any], param_types: tp.List[str]
    ) -> tp.List[tp.List[str]]:

        rows = toolz.mapcat(
            lambda t: self.iter_rows(path=(), param_type=t[0], params=t[1]),
            variables.items(),
        )
        rows = toolz.groupby(lambda t: t[0][:depth], rows)

        def get_path_param_types(
            inputs: tp.Tuple,
        ) -> tp.List[str]:
            path, values = inputs
            param_type_params = toolz.groupby(lambda t: t[1], values)
            param_type_params = {
                # group_values = [ (path, param_type, param) ]
                param_type: [t[2] for t in group_values]  # select only params
                for param_type, group_values in param_type_params.items()
            }

            row = list(
                map(
                    lambda param_type: utils.format_count_and_size(
                        param_type_params.get(param_type, None)
                    ),
                    param_types,
                )
            )

            return ["/".join(path)] + row

        rows = map(get_path_param_types, rows.items())
        rows = toolz.sorted(rows, key=lambda t: t[0])

        return list(rows)

    def iter_rows(
        self, path: tp.Tuple[str, ...], param_type: str, params: tp.Mapping[str, tp.Any]
    ) -> tp.Iterable[tp.Tuple[tp.Tuple[str, ...], str, tp.Any]]:

        for name, param in params.items():
            if isinstance(param, tp.Mapping):
                yield from self.iter_rows(
                    path=path + (name,),
                    param_type=param_type,
                    params=param,
                )
            else:
                yield (path + (name,), param_type, param)
