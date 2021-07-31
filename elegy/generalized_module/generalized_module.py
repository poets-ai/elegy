import typing as tp
from abc import ABC, abstractmethod

import jax
from elegy import types, utils
from rich.table import Table
from rich.text import Text

REGISTRY: tp.Dict[tp.Type, tp.Type["GeneralizedModule"]] = {}


class ModuleExists(Exception):
    pass


class GeneralizedModule(ABC):
    @abstractmethod
    def __init__(self, module: tp.Any):
        ...

    @abstractmethod
    def init(self, rng: types.RNGSeq) -> tp.Callable[..., types.OutputStates]:
        ...

    @abstractmethod
    def apply(
        self,
        params: tp.Any,
        states: tp.Any,
        training: bool,
        rng: types.RNGSeq,
    ) -> tp.Callable[..., types.OutputStates]:
        ...

    def update(
        self,
        params: tp.Optional[types.ModuleParams],
        states: tp.Optional[types.ModuleStates],
    ):
        ...

    @abstractmethod
    def summary(
        self,
        x: tp.Any,
        depth: int,
        run_eagerly: bool,
        eval_shape: bool,
    ) -> str:
        ...


class CallableModule(GeneralizedModule):
    def __init__(self, f: tp.Callable):
        self.f = f

    def init(self, rng: types.RNGSeq) -> tp.Callable[..., types.OutputStates]:
        def lambda_(*args, **kwargs) -> types.OutputStates:

            output = utils.inject_dependencies(self.f)(*args, **kwargs)

            if isinstance(output, types.OutputStates):
                return output
            else:
                return types.OutputStates(
                    preds=output,
                    params=None,
                    states=None,
                )

        return lambda_

    def apply(
        self,
        params: tp.Any,
        states: tp.Any,
        training: bool,
        rng: types.RNGSeq,
    ) -> tp.Callable[..., types.OutputStates]:
        def lambda_(*args, **kwargs) -> types.OutputStates:

            output = utils.inject_dependencies(self.f)(*args, **kwargs)

            if isinstance(output, types.OutputStates):
                return output
            else:
                return types.OutputStates(
                    preds=output,
                    params=None,
                    states=None,
                )

        return lambda_

    def summary(
        self,
        x: tp.Any,
        depth: int,
        run_eagerly: bool,
        eval_shape: bool,
    ) -> str:
        rng = types.RNGSeq(42)

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
        params_table = Table(
            title="Parameters",
            show_header=True,
            show_lines=True,
            title_justify="left",
            # show_footer=True,
            # box=rich.box.HORIZONTALS,
        )

        variables = {"Parameters": params, "States": states}

        params_table.add_column("")

        for param_type in variables:
            params_table.add_column(param_type)

        params_rows = []

        # add totals row
        totals = ["Total"] + [
            utils.format_count_and_size(params) for params in variables.values()
        ]
        params_rows.append(totals)

        utils.add_padding(params_rows)
        params_rows[0][0] = Text.from_markup(params_rows[0][0], justify="right")

        for row in params_rows:
            params_table.add_row(*row)

        # all params total as caption
        params_table.caption_style = "bold"
        params_table.caption = "\nTotal Parameters: " + utils.format_count_and_size(
            variables, add_padding=False
        )

        summary += "\n" + utils.get_table_repr(params_table)

        return summary


def register_module_for(
    module_type,
) -> tp.Callable[[tp.Type[GeneralizedModule]], tp.Any]:
    def wrapper(generalized_module_type: tp.Type[GeneralizedModule]) -> tp.Any:
        if module_type in REGISTRY:
            raise ModuleExists(
                f"Type {module_type} already registered with {REGISTRY[module_type]}"
            )

        REGISTRY[module_type] = generalized_module_type

        return generalized_module_type

    return wrapper


def generalize(
    module: tp.Any,
    callable_default: tp.Type[GeneralizedModule] = CallableModule,
) -> GeneralizedModule:

    if isinstance(module, GeneralizedModule):
        return module

    generalized_module_type: tp.Optional[tp.Type[GeneralizedModule]] = None

    for module_type in REGISTRY:
        if isinstance(module, module_type):
            generalized_module_type = REGISTRY[module_type]

    if generalized_module_type is None:
        if isinstance(module, tp.Callable):
            return callable_default(module)
        else:
            raise ValueError(f"No GeneralizedModule found for {module}.")

    return generalized_module_type(module)


def is_generalizable(module: tp.Any, accept_callable: bool = False) -> bool:

    if isinstance(module, GeneralizedModule):
        return True

    for module_type in REGISTRY:
        if isinstance(module, module_type):
            return True

    if accept_callable and isinstance(module, tp.Callable):
        return True

    return False
