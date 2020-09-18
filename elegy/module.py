from elegy.frozen_dict import FrozenDict
import functools
import threading
import typing as tp
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from io import StringIO

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from tabulate import tabulate

from elegy import types, utils
from elegy.random import RNG
from elegy.utils import EMPTY, Empty, Mode, ModuleOrderError

__all__ = [
    "Module",
    "to_module",
    "add_loss",
    "add_metric",
    "add_summary",
    "next_rng_key",
]

T = tp.TypeVar("T")


class DynamicContext(tp.NamedTuple):
    rng: RNG


class StaticContext(tp.NamedTuple):
    training: bool
    initializing: bool
    losses: tp.Optional[FrozenDict[str, tp.Any]]
    metrics: tp.Optional[FrozenDict[str, tp.Any]]
    summaries: tp.Optional[
        tp.Tuple[tp.Tuple[tp.Optional["Module"], str, np.ndarray], ...]
    ]
    names: tp.Optional[tp.Tuple[str, ...]]
    level_names: tp.Optional[tp.Tuple[str, ...]]
    module: tp.Optional["Module"]
    inside_call: tp.Optional[bool]
    module_index: tp.Optional[int]


class LocalContext(utils.Protocol):
    rng: RNG
    training: bool
    initializing: bool
    losses: tp.Optional[tp.Dict[str, tp.Any]]
    metrics: tp.Optional[tp.Dict[str, tp.Any]]
    summaries: tp.Optional[tp.List[tp.Tuple[tp.Optional["Module"], str, np.ndarray]]]
    names: tp.Optional[tp.List[str]]
    level_names: tp.Optional[tp.List[str]]
    module: tp.Optional["Module"]
    inside_call: tp.Optional[bool]
    module_index: tp.Optional[int]

    def dynamic_context(self) -> "DynamicContext":
        ...

    def static_context(self) -> "StaticContext":
        ...

    def set_from(self, statics: "StaticContext", dynamics: "DynamicContext"):
        ...


@dataclass
class _LocalContext(threading.local):
    rng: RNG
    training: bool
    initializing: bool
    losses: tp.Optional[tp.Dict[str, tp.Any]]
    metrics: tp.Optional[tp.Dict[str, tp.Any]]
    summaries: tp.Optional[tp.List[tp.Tuple[tp.Optional["Module"], str, np.ndarray]]]
    names: tp.Optional[tp.List[str]]
    level_names: tp.Optional[tp.List[str]]
    module: tp.Optional["Module"]
    inside_call: tp.Optional[bool]
    module_index: tp.Optional[int]

    def dynamic_context(self) -> "DynamicContext":
        return DynamicContext(rng=self.rng)

    def static_context(self) -> "StaticContext":
        return StaticContext(
            training=self.training,
            initializing=self.initializing,
            losses=FrozenDict(self.losses) if self.losses is not None else None,
            metrics=FrozenDict(self.metrics) if self.metrics is not None else None,
            summaries=tuple(self.summaries) if self.summaries is not None else None,
            names=tuple(self.names) if self.names is not None else None,
            level_names=tuple(self.level_names)
            if self.level_names is not None
            else None,
            module=self.module,
            inside_call=self.inside_call,
            module_index=self.module_index,
        )

    def set_from(self, static: "StaticContext", dynamic: "DynamicContext"):
        # dynamic
        self.rng = dynamic.rng

        # static
        self.training = static.training
        self.initializing = static.initializing
        self.losses = static.losses.unfreeze() if static.losses is not None else None
        self.metrics = static.metrics.unfreeze() if static.metrics is not None else None
        self.summaries = (
            list(static.summaries) if static.summaries is not None else None
        )
        self.names = list(static.names) if static.names is not None else None
        self.level_names = (
            list(static.level_names) if static.level_names is not None else None
        )
        self.module = static.module
        self.inside_call = static.inside_call
        self.module_index = static.module_index


LOCAL: LocalContext = _LocalContext(
    rng=RNG(42),
    training=True,
    initializing=False,
    losses=None,
    metrics=None,
    summaries=None,
    names=None,
    level_names=None,
    module=None,
    inside_call=None,
    module_index=None,
)


def context(
    rng: tp.Optional[RNG] = None,
    training: tp.Optional[bool] = None,
    hooks: bool = False,
    summaries: bool = False,
) -> tp.ContextManager[LocalContext]:

    prev_rng = LOCAL.rng
    prev_training = LOCAL.training
    prev_losses = LOCAL.losses
    prev_metrics = LOCAL.metrics
    prev_summaries = LOCAL.summaries
    prev_names = LOCAL.names
    prev_level_names = LOCAL.level_names

    LOCAL.rng = rng if rng is not None else LOCAL.rng
    LOCAL.training = training if training is not None else LOCAL.training
    LOCAL.losses = {} if hooks else LOCAL.losses
    LOCAL.metrics = {} if hooks else LOCAL.metrics
    LOCAL.summaries = [] if summaries else LOCAL.summaries
    LOCAL.names = [] if hooks or summaries else LOCAL.names
    LOCAL.level_names = [] if hooks or summaries else LOCAL.level_names

    try:
        yield _LocalContext(**vars(LOCAL))
    finally:
        # # clean
        # if LOCAL.level_names is not None:
        #     LOCAL.level_names.clear()

        # revert
        LOCAL.rng = prev_rng
        LOCAL.training = prev_training
        LOCAL.losses = prev_losses
        LOCAL.metrics = prev_metrics
        LOCAL.summaries = prev_summaries
        LOCAL.names = prev_names
        LOCAL.level_names = prev_level_names


context = contextmanager(context)


def construct_module(cls, *args, **kwargs) -> "Module":
    module: Module = cls.__new__(cls, *args, **kwargs)
    with instantiation_context(module):
        cls.__init__(module, *args, **kwargs)

    assert module is not None

    if (
        not hasattr(module, "name")
        or not hasattr(module, "_params")
        or not hasattr(module, "_states")
        or not hasattr(module, "_submodules")
    ):
        raise ValueError(
            "Constructing a Module without calling the super constructor "
            "is not supported."
        )

    for key, value in vars(module).items():
        if not key.startswith("_") and leaf_isinstance(value, Module):
            module._submodules.append(key)

    return module


class ModuleMeta(ABCMeta):
    def __call__(cls: tp.Type[T], *args, **kwargs) -> "Module":

        # Set unique on parent when using inside `call`
        if LOCAL.inside_call:
            assert LOCAL.module_index is not None
            assert LOCAL.module

            index = LOCAL.module_index
            parent = LOCAL.module

            if len(parent._dynamic_submodules) > index:
                module = getattr(
                    parent,
                    parent._dynamic_submodules[index],
                )
                assert isinstance(module, Module)

                if not isinstance(module, cls):
                    raise ModuleOrderError(
                        f"Error retrieving module, expected type {cls.__name__}, got {module}. "
                        "This is probably due to control flow, you must guarantee that the same amount "
                        "of submodules will be created every time and that their order is the same."
                    )
            else:
                if not LOCAL.initializing:
                    raise ValueError(
                        f"Trying to create module of type'{cls.__name__}' outside of `init`."
                    )

                module = construct_module(cls, *args, **kwargs)

                name = get_unique_name(set(vars(parent)), module.name)
                setattr(parent, name, module)
                parent._submodules.append(name)
                parent._dynamic_submodules.append(name)

            LOCAL.module_index += 1

            return module
        else:
            return construct_module(cls, *args, **kwargs)


class Module(metaclass=ModuleMeta):
    """
    Basic Elegy Module. Its a thin wrapper around `hk.Module` that
    add custom functionalities related to Elegy.
    """

    name: str
    dtype: np.dtype
    _initialized: bool
    _params: tp.Dict[str, bool]
    _states_initial: tp.List[str]
    _submodules: tp.List[str]
    _dynamic_submodules: tp.List[str]
    _trainable: bool

    __all__ = [
        "__init__",
        "call",
        "init",
        "apply",
        "reset",
        "get_parameters",
        "set_parameters",
        "set_states",
        "submodules",
    ]

    def __init__(self, name: tp.Optional[str] = None, dtype: np.dtype = jnp.float32):
        """
        Initializes the current module with the given name.

        Subclasses should call this constructor before creating other modules or
        variables such that those modules are named correctly.

        Arguments:
            name: An optional string name for the class. Must be a valid elsePython
                identifier. If ``name`` is not provided then the class name for the
                current instance is converted to ``lower_snake_case`` and used instead.
        """
        self.name = name if name else utils.lower_snake_case(self.__class__.__name__)
        self.dtype = dtype
        self._params = {}
        self._states = []
        self._submodules = []
        self._dynamic_submodules = []
        self._initialized = False
        self._trainable = True

        utils.wraps(self.call)(self)
        self.jit = jit(self)
        self.init_jit = jit(self.init, modules=self)

    @property
    def initialized(self) -> bool:
        return self._initialized

    @initialized.setter
    def initialized(self, value: bool):
        tree_exec(lambda module: self._set_initialized(module, value), self)

    @staticmethod
    def _set_initialized(module: "Module", value: bool):
        module._initialized = value

    @property
    def trainable(self) -> bool:
        return self._trainable

    @trainable.setter
    def trainable(self, value: bool):
        tree_exec(lambda module: self._set_trainable(module, value), self)

    @staticmethod
    def _set_trainable(module: "Module", value: bool):
        module._trainable = value

    def __call__(self, *args, **kwargs) -> tp.Any:
        """
        Forwards all input arguments to the Module's `call` method and calls
        `elegy.add_summary` on the outputs.
        """
        with call_context(self):
            outputs = self.call(*args, **kwargs)

            add_summary(self, outputs)

            return outputs

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        utils.wraps(cls.call)(cls.init)

    @abstractmethod
    def call(self, *args, **kwargs):
        ...

    @property
    def submodules(self) -> tp.Dict[str, tp.Any]:
        """
        A dictionary with all submodules contained in this Module.
        """
        return {name: getattr(self, name) for name in self._submodules}

    def init(self, *args, **kwargs) -> None:
        """
        Initializes the module.

        Arguments:
            x: sample inputs.
        """

        with init_context():
            self(*args, **kwargs)

        self.initialized = True

    def add_parameter(
        self,
        name: str,
        shape: tp.Sequence[int] = (),
        dtype: tp.Optional[np.dtype] = None,
        initializer: tp.Union[
            tp.Callable[[tp.Sequence[int], tp.Any], tp.Any], tp.Any
        ] = jnp.zeros,
        trainable: bool = True,
    ) -> np.ndarray:
        """
        A hook that lets you add a parameter to the current module. The parameter will only be created once
        during `init` and will reused afterwards.

        Arguments:
            name: The name of the parameter. It must be unique and no other field/property/method
                of the instance can have that name.
            shape: The shape of the parameter.
            dtype: The type of the parameter.
            initializer: A callable that takes in a shape and dtype and returns the initial value.

        Returns:
            The value of the parameter.
        """

        if not hasattr(self, name):
            if not is_initializing():
                raise ValueError(f"Trying to initialize '{name}' outside of `init`.")

            self._params[name] = trainable

            if dtype is None:
                dtype = self.dtype

            initial_value = (
                initializer(shape, dtype)
                if isinstance(initializer, tp.Callable)
                else initializer
            )

            setattr(self, name, initial_value)

        elif name not in self._params:
            raise ValueError(
                f"Class already contained a property named '{name}', "
                "please use a unique name for the parameter."
            )

        value = getattr(self, name)

        return value

    def update_parameter(self, name: str, value: tp.Any) -> None:
        """
        A hook that lets you update a state of the current module, if the state does not
        exist it will be created.

        Arguments:
            name: The name of the state. It must be unique and no other field/property/method
                of the instance can have that name.
            value: The updated value of the state.
        """

        if hasattr(self, name):
            if is_initializing():
                return

            setattr(self, name, value)
        else:
            raise ValueError(f"Parameter {name} not found in {self}.")

    def get_parameters(
        self,
        trainable: tp.Optional[bool] = None,
    ) -> types.Parameters:
        """
        Recursively collects a dictionary with the parameters of this module
        and all submodules within it.

        Returns:
        """

        params = module_tree_map(
            lambda module: {
                key: getattr(module, key)
                for key, param_is_trainable in module._params.items()
                if hasattr(module, key)
                and (
                    trainable is None
                    or trainable == (param_is_trainable and module.trainable)
                )
            },
            self,
        )
        assert isinstance(params, tp.Dict)
        return params

    def set_parameters(self, values: types.Parameters) -> None:
        """
        Recursively sets the parameters of this module
        and all submodules within it given a dictionary with a corresponding
        structure.
        """

        def f(module: Module, values: tp.Dict[str, tp.Any]):
            for key, value in values.items():
                if key in module._params:
                    setattr(module, key, value)

        tree_apply(f, self, values)

    def get_states(self) -> tp.Dict[str, tp.Any]:
        """
        Recursively collects a dictionary with the static (non-array) states of this module
        and all submodules within it.

        Returns:
        """

        params = module_tree_map(
            lambda module: dict(
                trainable=module._trainable,
            ),
            self,
        )
        assert isinstance(params, tp.Dict)
        return params

    def set_states(self, values: tp.Dict[str, tp.Any]) -> None:
        """
        Recursively sets the states (non-arrays) of this module
        and all submodules within it given a dictionary with a corresponding
        structure.
        """

        def f(module: Module, states):
            module._trainable = states["trainable"]

        tree_apply(f, self, values)

    def reset(self):
        """
        Recursively deletes the parameters and states of this module
        and all submodules within it.
        """

        def clear_module(module: Module):

            for name in module._params:
                delattr(module, name)

            for name in module._states:
                delattr(module, name)

            # module._params = set()
            # module._states = set()

        tree_exec(clear_module, self)

    def parameters_size(self, include_submodules: bool = True):
        if include_submodules:
            return sum(x.size for x in jax.tree_leaves(self.get_parameters()))
        else:
            return sum(
                x.size
                for x in jax.tree_leaves(
                    [getattr(self, key) for key in self._params if hasattr(self, key)]
                )
            )

    def states_size(self, include_submodules: bool = True):
        if include_submodules:
            return sum(
                x.size for x in jax.tree_leaves(self.get_parameters(trainable=False))
            )
        else:
            return sum(
                x.size
                for x in jax.tree_leaves(
                    [getattr(self, key) for key in self._states if hasattr(self, key)]
                )
            )

    def parameters_bytes(self, include_submodules: bool = True):
        if include_submodules:
            return sum(
                x.size * x.dtype.itemsize
                for x in jax.tree_leaves(self.get_parameters())
            )
        else:
            return sum(
                x.size * x.dtype.itemsize
                for x in jax.tree_leaves(
                    [getattr(self, key) for key in self._params if hasattr(self, key)]
                )
            )

    def states_bytes(self, include_submodules: bool = True):
        if include_submodules:
            return sum(
                x.size * x.dtype.itemsize
                for x in jax.tree_leaves(self.get_parameters(trainable=False))
            )
        else:
            return sum(
                x.size * x.dtype.itemsize
                for x in jax.tree_leaves(
                    [getattr(self, key) for key in self._states if hasattr(self, key)]
                )
            )

    def summary(
        self, x, depth: int = 2, tablefmt: str = "fancy_grid", **tablulate_kwargs
    ):
        """
        Prints a summary of the network.

        Arguments:
            x: A sample of inputs to the network.
            depth: The level number of nested level which will be showed.
                Information about summaries from modules deeper than `depth`
                will be aggregated together.
            tablefmt: A string represeting the style of the table generated by
                `tabulate`. See
                [python-tabulate](https://github.com/astanin/python-tabulate)
                for more options.
            tablulate_kwargs: Additional keyword arguments passed to `tabulate`.
                See [python-tabulate](https://github.com/astanin/python-tabulate)
                for more options.
        """
        if not self.initialized:
            self.init(x)

        with context(summaries=True):
            self(x)

            summaries = get_summaries()

        assert summaries is not None

        def format_output(outputs) -> str:
            file = StringIO()
            outputs = jax.tree_map(lambda x: f"{x.shape}{{pad}}  {x.dtype}", outputs)
            yaml.safe_dump(
                outputs, file, default_flow_style=False, indent=2, explicit_end=False
            )
            return file.getvalue().replace("\n...", "")

        def format_size(size):
            return (
                f"{size / 1e9 :,.1f} GB"
                if size > 1e9
                else f"{size / 1e6 :,.1f} MB"
                if size > 1e6
                else f"{size / 1e3 :,.1f} KB"
                if size > 1e3
                else f"{size:,} B"
            )

        table: tp.List = [["Inputs", format_output(x), "0", "0"]]

        for module, base_name, value in summaries:
            base_name_parts = base_name.split("/")[1:]
            module_depth = len(base_name_parts)

            if module_depth > depth:
                continue

            include_submodules = module_depth == depth

            params_count = (
                module.parameters_size(include_submodules) if module is not None else 0
            )
            params_size = (
                module.parameters_bytes(include_submodules) if module is not None else 0
            )
            states_count = (
                module.states_size(include_submodules) if module is not None else 0
            )
            states_size = (
                module.states_bytes(include_submodules) if module is not None else 0
            )
            class_name = f"({module.__class__.__name__})" if module is not None else ""

            base_name = "/".join(base_name_parts)

            if not base_name:
                base_name = "Outputs"

            table.append(
                [
                    f"{base_name}{{pad}}  {class_name}",
                    format_output(value),
                    f"{params_count:,}{{pad}}    {format_size(params_size)}"
                    if params_count > 0
                    else "0",
                    f"{states_count:,}{{pad}}    {format_size(states_size)}"
                    if states_count > 0
                    else "0",
                ]
            )

        # add papdding
        for col in range(4):
            max_length = max(
                len(line.split("{pad}")[0])
                for row in table
                for line in row[col].split("\n")
            )

            for row in table:
                row[col] = "\n".join(
                    line.format(
                        pad=" " * (max_length - len(line.rstrip().split("{pad}")[0]))
                    )
                    for line in row[col].rstrip().split("\n")
                )

        print(
            "\n"
            + tabulate(
                table,
                headers=[
                    "Layer",
                    "Outputs Shape",
                    "Trainable\nParameters",
                    "Non-trainable\nParameters",
                ],
                tablefmt=tablefmt,
                **tablulate_kwargs,
            )
        )

        params_count = self.parameters_size()
        params_size = self.parameters_bytes()
        states_count = self.states_size()
        states_size = self.states_bytes()
        total_count = params_count + states_count
        total_size = params_size + states_size

        print(
            tabulate(
                [
                    [
                        f"Total Parameters:",
                        f"{total_count:,}",
                        f"{format_size(total_size)}" if total_count > 0 else "",
                    ],
                    [
                        f"Trainable Parameters:",
                        f"{params_count:,}",
                        f"{format_size(params_size)}" if params_count > 0 else "",
                    ],
                    [
                        f"Non-trainable Parameters:",
                        f"{states_count:,}",
                        f"{format_size(states_size)}" if states_count > 0 else "",
                    ],
                ],
                tablefmt="plain",
            )
            + "\n"
        )


# -------------------------------------------------------------
# hooks
# -------------------------------------------------------------


def get_module() -> tp.Optional[Module]:
    return LOCAL.module


def get_rng() -> RNG:
    return LOCAL.rng


def set_rng(rng: RNG) -> None:
    LOCAL.rng = rng


def next_rng_key() -> jnp.ndarray:
    return LOCAL.rng()


def is_initializing() -> bool:
    return LOCAL.initializing


def set_training(training: bool) -> None:
    LOCAL.training = training


def is_training() -> bool:
    return LOCAL.training


def get_losses() -> tp.Optional[tp.Dict[str, tp.Any]]:
    return LOCAL.losses


def get_metrics() -> tp.Optional[tp.Dict[str, tp.Any]]:
    return LOCAL.metrics


def get_summaries() -> tp.Optional[
    tp.List[tp.Tuple[tp.Optional["Module"], str, np.ndarray]]
]:
    return LOCAL.summaries


def base_name() -> str:
    return "/".join(LOCAL.names) if LOCAL.names is not None else ""


@contextmanager
def rng_context(rng: RNG):
    current_rng = LOCAL.rng
    LOCAL.rng = rng

    try:
        yield
    finally:
        LOCAL.rng = current_rng


@contextmanager
def training_context(training: bool):
    current_training = LOCAL.training
    LOCAL.training = training

    try:
        yield
    finally:
        LOCAL.training = current_training


@contextmanager
def name_context(name: str) -> tp.Iterator[str]:

    if LOCAL.names is None or LOCAL.level_names is None:
        yield ""
        return

    current_level_names = LOCAL.level_names

    name = get_unique_name(set(current_level_names), name)

    current_level_names.append(name)  # add name to current level
    LOCAL.names.append(name)
    LOCAL.level_names = []  # create new level for children

    try:
        yield name
    finally:
        LOCAL.names.pop()
        LOCAL.level_names = current_level_names


@contextmanager
def call_context(module: Module):

    prev_module = LOCAL.module
    prev_inside_call = LOCAL.inside_call
    prev_module_index = LOCAL.module_index

    LOCAL.module = module
    LOCAL.inside_call = True
    LOCAL.module_index = 0

    with name_context(module.name):

        try:
            yield
        finally:
            LOCAL.module = prev_module
            LOCAL.inside_call = prev_inside_call
            LOCAL.module_index = prev_module_index


@contextmanager
def instantiation_context(module: Module):

    prev_module = LOCAL.module
    prev_inside_call = LOCAL.inside_call

    LOCAL.inside_call = False

    try:
        yield
    finally:
        LOCAL.module = prev_module
        LOCAL.inside_call = prev_inside_call


def add_summary(module_or_name: tp.Union[Module, str], value: np.ndarray) -> None:
    """
    A hook that lets you define a summary in the current module. Its primary
    use is to keep track of certain values as they flow through the network
    so `Model.summary()` can show a representation of architecture.

    ```python
    def call(self, x):
        ...
        y = jax.nn.relu(x)
        elegy.add_summary("relu", y)
        ...
    ```

    The summaries will be aggregated by [`apply`][elegy.module.Module.apply]
    if `get_summaries` is set to `True`, else this hook does nothing.

    ```python
    transformed_state = transform.apply(..., get_summaries=True, ...)
    ```

    Arguments:
        name: The name of the loss. If a summary with the same
            `name` already exists a unique identifier will be generated.
        value: The value for the summary.
    """

    if LOCAL.summaries is None:
        return

    name = base_name()

    if isinstance(module_or_name, str):
        name = f"{name}/{module_or_name}" if name else module_or_name
        name = get_unique_name({t[1] for t in LOCAL.summaries}, name)
        module = None
    else:
        module = module_or_name

    LOCAL.summaries.append((module, name, value))


def add_loss(name: str, value: np.ndarray) -> None:
    """
    A hook that lets you define a loss within a [`module`][elegy.module.Module].

    ```python
    w = self.add_parameter("w", [3, 5], initializer=jnp.ones)

    # L2 regularization penalty
    elegy.add_loss("l2_regularization", 0.01 * jnp.mean(w ** 2))
    ```

    The loss will be aggregated by [`Module.apply`][elegy.module.Module.apply]
    and automatically handled by [`Model`][elegy.model.Model].

    Arguments:
        name: The name of the loss. If a `name` is repeated on
            different calls values will be added together.
        value: The value for the loss.
    """
    if LOCAL.losses is None:
        return

    if not name.endswith("loss"):
        name += "_loss"

    if name in LOCAL.losses:
        LOCAL.losses[name] += value
    else:
        LOCAL.losses[name] = value


def add_metric(name: str, value: np.ndarray) -> None:
    """
    A hook that lets you define a metric within a [`module`][elegy.module.Module].

    ```python
    y = jax.nn.relu(x)
    elegy.add_metric("activation_mean", jnp.mean(y))
    ```

    The metrics will be aggregated by [`Module.apply`][elegy.module.Module.apply]
    and automatically handled by [`Model`][elegy.model.Model].

    Arguments:
        name: The name of the loss. If a metric with the same
            `name` already exists a unique identifier will be generated.
        value: The value for the metric.
    """
    if LOCAL.metrics is None:
        return

    name = f"{base_name()}/{name}"
    name = get_unique_name(set(LOCAL.metrics), name)
    LOCAL.metrics[name] = value


def init_context() -> tp.ContextManager:
    prev_initializing = LOCAL.initializing

    LOCAL.initializing = True

    try:
        yield
    finally:
        LOCAL.initializing = prev_initializing


init_context = contextmanager(init_context)


def get_dynamic_context() -> "DynamicContext":
    return LOCAL.dynamic_context()


def get_static_context() -> "StaticContext":
    return LOCAL.static_context()


def set_context(static: "StaticContext", dynamic: "DynamicContext"):
    LOCAL.set_from(static, dynamic)


# -------------------------------------------------------------
# transforms
# -------------------------------------------------------------


def jit(
    f: tp.Union[tp.Callable, Module],
    modules: tp.Optional[tp.Union[Module, tp.List[Module]]] = None,
    **kwargs,
):
    static_argnums = tuple(kwargs.pop("static_argnums", ()))

    if modules is None:
        modules = []
    elif isinstance(modules, Module):
        modules = [modules]

    if isinstance(f, Module):

        if modules is not None and f not in modules:
            modules.append(f)
        elif modules is None:
            modules = [f]

    if len(modules) < 1:
        raise ValueError("No module specified")

    static_argnums = (0, 1) + tuple(i + 4 for i in static_argnums)

    def jit_fn(
        states_tuple: tp.Tuple[FrozenDict[str, tp.Any], ...],
        statics: StaticContext,
        dynamics: DynamicContext,
        parameters_tuple: tp.Tuple[tp.Dict, ...],
        *args,
    ) -> tp.Tuple[tp.Any, StaticContext, DynamicContext, tp.Tuple]:
        assert isinstance(modules, list)

        # states_tuple is not set because its static, therefore no need to propagate down

        # set global state
        set_context(statics, dynamics)

        # set params to modules
        for module, parameters in zip(modules, parameters_tuple):
            module.set_parameters(parameters)

        outputs = f(*args)

        parameters_tuple = tuple(module.get_parameters() for module in modules)

        return (
            outputs,
            get_static_context(),
            get_dynamic_context(),
            parameters_tuple,
        )

    jit_fn = jax.jit(jit_fn, static_argnums, **kwargs)

    @functools.wraps(f)
    def wrapper(*args):
        assert isinstance(modules, list)

        states_tuple = utils.to_static(
            tuple(FrozenDict(module.get_states()) for module in modules)
        )
        statics = get_static_context()
        dynamics = get_dynamic_context()
        parameters_tuple = tuple(module.get_parameters() for module in modules)
        # static_argnums

        outputs, statics, dynamics, parameters_tuple = jit_fn(
            states_tuple,
            statics,
            dynamics,
            parameters_tuple,
            *args,
        )

        # set global state
        set_context(statics, dynamics)

        # set params to modules
        for module, parameters in zip(modules, parameters_tuple):
            module.set_parameters(parameters)

        return outputs

    return wrapper


def get_trainable_parameters(modules: tp.List[Module]) -> tp.List[tp.Any]:
    return [module.get_parameters(trainable=True) for module in modules]


def value_and_grad(
    f: tp.Union[tp.Callable, Module],
    modules: tp.Optional[tp.Union[Module, tp.List[Module]]] = None,
    parameters_fn: tp.Callable[
        [tp.List[Module]], tp.List[tp.Any]
    ] = get_trainable_parameters,
    **kwargs,
):
    is_list = isinstance(modules, tp.List)

    if modules is None:
        modules = []
    elif isinstance(modules, Module):
        modules = [modules]

    if isinstance(f, Module):

        if modules is not None and f not in modules:
            modules.append(f)
        elif modules is None:
            modules = [f]

    assert len(modules) > 0

    def grad_fn(parameters_tuple: tp.Tuple[tp.Dict, ...], *args, **kwargs):
        assert isinstance(parameters_tuple, tuple)
        assert isinstance(modules, list)

        # set traced parameters
        for module, parameters in zip(modules, parameters_tuple):
            module.set_parameters(parameters)

        outputs = f(*args, **kwargs)

        loss = outputs[0] if isinstance(outputs, tuple) else outputs

        parameters_tuple = tuple(module.get_parameters() for module in modules)

        return loss, (
            outputs,
            get_static_context(),
            get_dynamic_context(),
            parameters_tuple,
        )

    kwargs["has_aux"] = True
    grad_fn = jax.value_and_grad(grad_fn, **kwargs)

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        assert isinstance(modules, list)

        parameters_tuple = tuple(parameters_fn(modules))

        (_loss, (outputs, statics, dynamics, parameters_tuple)), grads = grad_fn(
            parameters_tuple, *args, **kwargs
        )

        parameters_tuple

        # set global state
        set_context(statics, dynamics)

        # set original untraced parameters
        for module, parameters in zip(modules, parameters_tuple):
            module.set_parameters(parameters)

        if not is_list:
            grads = grads[0]

        return outputs, grads

    return wrapper


# ------------------------------------------------------------------------
# utils
# ------------------------------------------------------------------------


def module_tree_map(
    f: tp.Callable[[Module], tp.Dict[str, tp.Any]],
    module: tp.Union[Module, tp.List, tp.Tuple, tp.Dict],
) -> tp.Union[tp.List, tp.Tuple, tp.Dict]:

    if isinstance(module, tp.List):
        return [module_tree_map(f, module) for module in module]
    elif isinstance(module, tp.Tuple):
        return tuple(module_tree_map(f, module) for module in module)
    elif isinstance(module, tp.Dict):
        return {key: module_tree_map(f, module) for key, module in module.items()}
    elif isinstance(module, Module):

        node = f(module)

        for submodule in module._submodules:
            value = module_tree_map(f, getattr(module, submodule))
            # if value:  # drop if empty
            node[submodule] = value

        return node
    else:
        return ()


def tree_apply(
    f: tp.Callable[[Module, tp.Dict[str, tp.Any]], None],
    module: tp.Union[Module, tp.List, tp.Tuple, tp.Dict],
    values: tp.Union[tp.List, tp.Tuple, tp.Dict],
):

    if isinstance(module, tp.List):
        assert isinstance(values, tp.List)

        for module, value in zip(module, values):
            tree_apply(f, module, value)

    elif isinstance(module, tp.Tuple):
        assert isinstance(values, tp.Tuple)

        for module, value in zip(module, values):
            tree_apply(f, module, value)

    elif isinstance(module, tp.Dict):
        assert isinstance(values, tp.Dict)

        for key, value in values.items():
            tree_apply(f, module[key], value)

    elif isinstance(module, Module):
        assert isinstance(values, tp.Dict)

        f(module, values)

        for key, value in values.items():
            if key in module._submodules:
                tree_apply(f, getattr(module, key), value)


def tree_exec(
    f: tp.Callable[[Module], tp.Any],
    module: tp.Union[Module, tp.List, tp.Tuple, tp.Dict],
):

    if isinstance(module, tp.List):

        for module in module:
            tree_exec(f, module)

    elif isinstance(module, tp.Tuple):

        for module in module:
            tree_exec(f, module)

    elif isinstance(module, tp.Dict):

        for key, value in module.items():
            tree_exec(f, module[key])

    elif isinstance(module, Module):
        f(module)

        for key in module._submodules:
            tree_exec(f, getattr(module, key))


def leaf_isinstance(obj: tp.Any, types) -> tp.Type:

    if isinstance(obj, (tp.List, tp.Tuple)) and obj:
        return any(leaf_isinstance(elem, types) for elem in obj)
    elif isinstance(obj, tp.Dict) and obj:
        return any(leaf_isinstance(elem, types) for elem in obj.values())
    else:
        return isinstance(obj, types)


def get_unique_name(
    names: tp.Set[str],
    name: str,
):

    if name not in names:
        return name

    i = 1
    while f"{name}_{i}" in names:
        i += 1

    return f"{name}_{i}"


def to_module(f):
    class MyModule(Module):
        def __init__(self, name: tp.Optional[str] = None):
            super().__init__(
                name=utils.lower_snake_case(f.__name__) if name is None else name
            )
            self.call = f

        def call(self, *args, **kwargs):
            ...

    MyModule.__name__ = f.__name__

    return MyModule


def as_initial(name):
    return f"{name}__initial__"
