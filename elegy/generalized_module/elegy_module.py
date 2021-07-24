import typing as tp

import jax
import toolz

from elegy import hooks, types, utils
from elegy.module import Module
from rich.table import Table
from rich.text import Text

from .generalized_module import GeneralizedModule, is_generalizable, register_module_for


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
        """
        Prints a summary of the network.
        Arguments:
            x: A sample of inputs to the network.
            depth: The level number of nested level which will be showed.
                Information about summaries from modules deeper than `depth`
                will be aggregated together.
        """

        entries: tp.List[types.SummaryTableEntry]

        # TODO: try to cache the jitted function
        method = self.summary_step if run_eagerly else jax.jit(self.summary_step)

        if eval_shape:
            entries = jax.eval_shape(
                self.summary_step,
                x_args,
                x_kwargs,
                params,
                states,
                rng,
            )
        else:
            entries = method(
                x_args,
                x_kwargs,
                params,
                states,
                rng,
            )

        total_entry = entries[-1]
        entries = entries[:-1]

        depth_groups: tp.Dict[str, tp.List[types.SummaryTableEntry]] = toolz.groupby(
            lambda entry: "/".join(entry.path.split("/")[:depth]), entries
        )

        entries = [
            utils.get_grouped_entry(entry, depth_groups)
            for entry in entries
            if entry.path in depth_groups
        ]

        main_table = Table(
            show_header=True,
            show_lines=True,
            show_footer=True,
            # box=rich.box.HORIZONTALS,
        )

        main_table.add_column("Layer")
        main_table.add_column("Outputs Shape")
        main_table.add_column("Trainable\nParameters")
        main_table.add_column("Non-trainable\nParameters")

        rows: tp.List[tp.List[str]] = []

        rows.append(["Inputs", utils.format_output(x), "", ""])

        for entry in entries:
            rows.append(
                [
                    f"{entry.path}{{pad}}  "
                    + (
                        f"[dim]{entry.module_type_name}[/]"
                        if entry.module_type_name
                        else ""
                    ),
                    utils.format_output(entry.output_value),
                    f"[green]{entry.trainable_params_count:,}[/]{{pad}}    {utils.format_size(entry.trainable_params_size)}"
                    if entry.trainable_params_count > 0
                    else "",
                    f"[green]{entry.non_trainable_params_count:,}[/]{{pad}}    {utils.format_size(entry.non_trainable_params_size)}"
                    if entry.non_trainable_params_count > 0
                    else "",
                ]
            )

        # global summaries
        params_count = total_entry.trainable_params_count
        params_size = total_entry.trainable_params_size
        states_count = total_entry.non_trainable_params_count
        states_size = total_entry.non_trainable_params_size
        total_count = params_count + states_count
        total_size = params_size + states_size

        rows.append(
            [
                "",
                "Total",
                (
                    f"[green]{params_count:,}[/]{{pad}}    {utils.format_size(params_size)}"
                    if params_count > 0
                    else ""
                ),
                (
                    f"[green]{states_count:,}[/]{{pad}}    {utils.format_size(states_size)}"
                    if states_count > 0
                    else ""
                ),
            ]
        )

        # add padding
        utils.add_padding(rows)

        for row in rows[:-1]:
            main_table.add_row(*row)

        main_table.columns[1].footer = Text.from_markup(rows[-1][1], justify="right")
        main_table.columns[2].footer = rows[-1][2]
        main_table.columns[3].footer = rows[-1][3]
        main_table.caption_style = "bold"
        main_table.caption = (
            "\nTotal Parameters: "
            + f"[green]{total_count:,}[/]   {utils.format_size(total_size)}"
            if total_count > 0
            else ""
        )

        summary = "\n" + utils.get_table_repr(main_table)

        return summary

    def summary_step(
        self,
        x_args: tp.Tuple,
        x_kwargs: tp.Dict[str, tp.Any],
        params: tp.Any,
        states: tp.Any,
        rng: types.RNGSeq,
    ) -> tp.List[types.SummaryTableEntry]:
        training = True

        with hooks.context(summaries=True):
            _, params, states = self.apply(
                params=params, states=states, training=training, rng=rng
            )(*x_args, **x_kwargs)

            summaries = hooks.get_summaries()

        entries: tp.List[types.SummaryTableEntry] = []

        for path, module, value in summaries:

            module_params, module_states = self.get_summary_params(
                path=path,
                module=module,
                value=value,
                params=params,
                states=states,
            )

            entries.append(
                types.SummaryTableEntry(
                    path=("/".join(map(str, path)) if path else "*"),
                    module_type_name=(
                        module.__class__.__name__ if is_generalizable(module) else ""
                    ),
                    output_value=value,
                    trainable_params_count=(
                        utils.parameters_count(module_params)
                        if module_params is not None
                        else 0
                    ),
                    trainable_params_size=(
                        utils.parameters_bytes(module_params)
                        if module_params is not None
                        else 0
                    ),
                    non_trainable_params_count=(
                        utils.parameters_count(module_states)
                        if module_states is not None
                        else 0
                    ),
                    non_trainable_params_size=(
                        utils.parameters_bytes(module_states)
                        if module_states is not None
                        else 0
                    ),
                )
            )

        entries.append(
            types.SummaryTableEntry.totals_entry(
                trainable_params_count=utils.parameters_count(params),
                trainable_params_size=utils.parameters_bytes(params),
                non_trainable_params_count=utils.parameters_count(states),
                non_trainable_params_size=utils.parameters_bytes(states),
            )
        )

        return entries

    def get_summary_params(
        self,
        path: types.Path,
        module: tp.Any,
        value: tp.Any,
        params: types.NetParams,
        states: types.NetStates,
    ) -> tp.Tuple[tp.Optional[types.Pytree], tp.Optional[types.Pytree]]:

        if params is None:
            params_tree = None
        else:
            params_tree = utils.get_path_params(path, params)
            # filter out submodules
            if params_tree is not None:
                assert isinstance(module, Module)
                params_tree = {
                    name: value
                    for name, value in params_tree.items()
                    if name in module._spec
                }

        if states is None:
            states_tree = None
        else:
            states_tree = {
                collection: utils.get_path_params(path, state)
                for collection, state in states.items()
            }
            # filter out submodules
            states_tree = {
                collection: {
                    name: value for name, value in state.items() if name in module._spec
                }
                for collection, state in states_tree.items()
                if state is not None
            }

        return params_tree, states_tree
