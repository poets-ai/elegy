from elegy.utils import EMPTY, Empty, ModuleOrderError
from elegy.random import RNG
import threading
import typing as tp
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager

# -------------------------------------------------------------
# context
# -------------------------------------------------------------
from dataclasses import dataclass

# -----------------------------------------------------------------
# RNG
# ----------------------------------------------------------------
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from elegy import types, utils

__all__ = [
    # "ApplyCallable",
    # "ApplyContext",
    # "Context",
    # "InitCallable",
    "Module",
    "to_module",
    "add_loss",
    "add_metric",
    "add_summary",
    "next_rng_key",
]

T = tp.TypeVar("T")


class LocalContext(utils.Protocol):
    rng: RNG
    training: bool
    initializing: bool
    losses: tp.Optional[tp.Dict[str, tp.Any]]
    metrics: tp.Optional[tp.Dict[str, tp.Any]]
    summaries: tp.Optional[tp.List[tp.Tuple[tp.Optional["Module"], str, np.ndarray]]]
    names: tp.Optional[tp.List[str]]
    level_names: tp.Optional[tp.List[tp.List[str]]]
    module: tp.Optional["Module"]
    inside_call: tp.Optional[bool]
    module_index: tp.Optional[int]


@dataclass
class _LocalContext(threading.local):
    rng: RNG
    training: bool
    initializing: bool
    losses: tp.Optional[tp.Dict[str, tp.Any]]
    metrics: tp.Optional[tp.Dict[str, tp.Any]]
    summaries: tp.Optional[tp.List[tp.Tuple[tp.Optional["Module"], str, np.ndarray]]]
    names: tp.Optional[tp.List[str]]
    level_names: tp.Optional[tp.List[tp.List[str]]]
    module: tp.Optional["Module"]
    inside_call: tp.Optional[bool]
    module_index: tp.Optional[int]


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


@contextmanager
def context(
    rng: tp.Union[Empty, RNG] = EMPTY,
    training: tp.Union[Empty, bool] = EMPTY,
    initializing: tp.Union[Empty, bool] = EMPTY,
    losses: tp.Union[Empty, tp.Dict[str, tp.Any]] = EMPTY,
    metrics: tp.Union[Empty, tp.Dict[str, tp.Any]] = EMPTY,
    summaries: tp.Union[
        Empty, tp.List[tp.Tuple[tp.Optional["Module"], str, np.ndarray]]
    ] = EMPTY,
    names: tp.Union[Empty, tp.List[str]] = EMPTY,
    level_names: tp.Union[Empty, tp.List[tp.List[str]]] = EMPTY,
    module: tp.Union[Empty, "Module"] = EMPTY,
    inside_call: tp.Union[Empty, bool] = EMPTY,
    module_index: tp.Union[Empty, int] = EMPTY,
) -> tp.Iterator[LocalContext]:

    global LOCAL
    current = LOCAL

    if not isinstance(metrics, Empty) or not isinstance(summaries, Empty):
        if isinstance(names, Empty):
            names = []

        if isinstance(level_names, Empty):
            level_names = [[]]

    LOCAL = _LocalContext(
        rng=rng if not isinstance(rng, Empty) else current.rng,
        training=training if not isinstance(training, Empty) else current.training,
        initializing=initializing
        if not isinstance(initializing, Empty)
        else current.initializing,
        losses=losses if not isinstance(losses, Empty) else None,
        metrics=metrics if not isinstance(metrics, Empty) else None,
        summaries=summaries if not isinstance(summaries, Empty) else None,
        names=names if not isinstance(names, Empty) else None,
        level_names=level_names if not isinstance(level_names, Empty) else None,
        module=module if not isinstance(module, Empty) else None,
        inside_call=inside_call if not isinstance(inside_call, Empty) else None,
        module_index=module_index if not isinstance(module_index, Empty) else None,
    )

    try:
        yield LOCAL
    finally:
        LOCAL = current


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
        if key not in module._ignore and leaf_isinstance(value, Module):
            module._submodules.append(key)

    utils.wraps(module.call)(module)

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
    _params: tp.List[str]
    _states: tp.List[str]
    _states_initial: tp.List[str]
    _submodules: tp.List[str]
    _dynamic_submodules: tp.List[str]
    _ignore: tp.List[str]

    __all__ = [
        "__init__",
        "call",
        "init",
        "apply",
        "reset",
        "get_parameters",
        "set_parameters",
        "get_states",
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
        self._params = []
        self._states = []
        self._submodules = []
        self._dynamic_submodules = []
        self._ignore = []

    def __call__(self, *args, **kwargs) -> tp.Any:
        """
        Forwards all input arguments to the Module's `call` method and calls
        `elegy.add_summary` on the outputs.
        """
        with call_context(self):
            outputs = self.call(*args, **kwargs)

            add_summary(self, outputs)

            return outputs

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

    # def apply(
    #     self,
    #     parameters: tp.Optional[tp.Dict] = None,
    #     states: tp.Optional[tp.Dict] = None,
    #     rng: tp.Optional[tp.Union[np.ndarray, int]] = None,
    #     get_summaries: bool = False,
    #     training: bool = True,
    # ) -> "ApplyCallable":
    #     """
    #     Applies your function injecting some context arguments.

    #     Arguments:
    #         parameters:
    #         states:
    #         rng:
    #         get_summaries:

    #     Returns:
    #         A function with the same signature as `call` that will
    #         execute the computation given the context arguments
    #         passed to `apply`.
    #     """

    #     @utils.wraps(self.call)
    #     def apply_fn(*args, **kwargs):

    #         current_parameters = self.get_parameters()
    #         current_states = self.get_states()

    #         assert current_parameters is not None
    #         assert current_states is not None

    #         if parameters is not None:
    #             self.set_parameters(parameters)

    #         if states is not None:
    #             self.set_states(states)

    #         with context(
    #             rng=rng,
    #             building=False,
    #             training=training,
    #             get_summaries=get_summaries,
    #         ) as ctx:
    #             outputs = self(*args, **kwargs)

    #         output_parameters = self.get_parameters()
    #         output_states = self.get_states()

    #         if parameters is not None:
    #             self.set_parameters(current_parameters)

    #         if states is not None:
    #             self.set_states(current_states)

    #         return (
    #             outputs,
    #             ApplyContext(
    #                 parameters=output_parameters,
    #                 states=output_states,
    #                 losses=ctx.losses,
    #                 metrics=ctx.metrics,
    #                 summaries=ctx.summaries,
    #             ),
    #         )

    #     return apply_fn

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
            if not module_initializing():
                raise ValueError(f"Trying to initialize '{name}' outside of `init`.")

            if trainable:
                self._params.append(name)
            else:
                self._states.append(name)

            if dtype is None:
                dtype = self.dtype

            initial_value = (
                initializer(shape, dtype)
                if isinstance(initializer, tp.Callable)
                else initializer
            )

            setattr(self, name, initial_value)

        elif name not in self._params + self._states:
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
            if module_initializing():
                return

            setattr(self, name, value)
        else:
            raise ValueError(f"Parameter {name} not found in {self}.")

    def get_parameters(self) -> types.Parameters:
        """
        Recursively collects a dictionary with the parameters of this module
        and all submodules within it.

        Returns:
        """
        params = module_tree_map(
            lambda module: {
                key: getattr(module, key)
                for key in getattr(module, "_params")
                if hasattr(module, key)
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
        set_tree(self, values, "_params")

    def get_states(self, _initial: bool = False) -> types.States:
        """
        Recursively collects a dictionary with the states of this module
        and all submodules within it.

        Returns:
        """
        states = module_tree_map(
            lambda module: {
                key: getattr(module, as_initial(key))
                if _initial
                else getattr(module, key)
                for key in getattr(module, "_states")
                if hasattr(module, key)
            },
            self,
        )
        assert isinstance(states, tp.Dict)
        return states

    def set_states(self, values: types.States):
        """
        Recursively sets the states of this module
        and all submodules within it given a dictionary with a corresponding
        structure.
        """
        set_tree(self, values, "_states")

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

    def clear_initial_states(self):
        def clear_module(module: Module):
            for name in module._states:
                delattr(module, as_initial(name))

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
            return sum(x.size for x in jax.tree_leaves(self.get_states()))
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
                x.size * x.dtype.itemsize for x in jax.tree_leaves(self.get_states())
            )
        else:
            return sum(
                x.size * x.dtype.itemsize
                for x in jax.tree_leaves(
                    [getattr(self, key) for key in self._states if hasattr(self, key)]
                )
            )


# -------------------------------------------------------------
# hooks
# -------------------------------------------------------------


@tp.overload
def rng() -> RNG:
    ...


@tp.overload
def rng(key: int) -> RNG:
    ...


@tp.overload
def rng(key: np.ndarray) -> RNG:
    ...


def rng(key: tp.Union[int, np.ndarray, Empty] = EMPTY) -> RNG:

    if not isinstance(key, Empty):
        LOCAL.rng = RNG(key)

    return LOCAL.rng


def next_rng_key() -> np.ndarray:
    return LOCAL.rng()


def module_initializing() -> bool:
    return LOCAL.initializing


@tp.overload
def training() -> bool:
    ...


@tp.overload
def training(status: bool) -> bool:
    ...


def training(status: tp.Union[bool, Empty] = EMPTY) -> bool:
    """
    A hook that gets/sets the current training status.

    ```python
    training = elegy.training()

    if training:
        ...
    else:
        ...
    ```

    Returns:
        A boolean value indicating whether training is currently happening.
    """

    if not isinstance(status, Empty):
        LOCAL.training = status

    return LOCAL.training


@tp.overload
def losses() -> tp.Optional[tp.Dict[str, tp.Any]]:
    ...


@tp.overload
def losses(
    initial: tp.Optional[tp.Dict[str, tp.Any]]
) -> tp.Optional[tp.Dict[str, tp.Any]]:
    ...


def losses(
    initial: tp.Union[tp.Dict[str, tp.Any], None, Empty] = EMPTY
) -> tp.Optional[tp.Dict[str, tp.Any]]:

    if not isinstance(initial, Empty):
        LOCAL.losses = initial

    return LOCAL.losses


@tp.overload
def metrics() -> tp.Optional[tp.Dict[str, tp.Any]]:
    ...


@tp.overload
def metrics(
    initial: tp.Optional[tp.Dict[str, tp.Any]]
) -> tp.Optional[tp.Dict[str, tp.Any]]:
    ...


def metrics(
    initial: tp.Union[tp.Dict[str, tp.Any], None, Empty] = EMPTY
) -> tp.Optional[tp.Dict[str, tp.Any]]:

    if not isinstance(initial, Empty):
        LOCAL.metrics = initial

    return LOCAL.metrics


def base_name() -> str:
    assert LOCAL.names
    return "/".join(LOCAL.names)


@contextmanager
def name_context(name: str) -> tp.Iterator[str]:

    if LOCAL.names is None or LOCAL.level_names is None:
        yield ""
        return

    name = get_unique_name(set(LOCAL.level_names[-1]), name)

    LOCAL.names.append(name)
    LOCAL.level_names[-1].append(name)  # add name to current level
    LOCAL.level_names.append([])  # create new level for children

    try:
        yield name
    finally:
        LOCAL.names.pop()
        LOCAL.level_names.pop()  #


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
        name = f"{name}/{module_or_name}"
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


@contextmanager
def init_context():
    prev_initializing = LOCAL.initializing

    LOCAL.initializing = True

    try:
        yield
    finally:
        LOCAL.initializing = prev_initializing


# ------------------------------------------------------------------------
# transform
# ------------------------------------------------------------------------


class ApplyCallable(utils.Protocol):
    def __call__(self, *args, **kwargs) -> tp.Tuple[tp.Any, "ApplyContext"]:
        ...


class InitCallable(utils.Protocol):
    def __call__(self, *args, **kwargs) -> tp.Tuple[types.Parameters, types.States]:
        ...


class ApplyContext(tp.NamedTuple):
    parameters: types.Parameters
    states: types.States
    losses: tp.Dict
    metrics: tp.Dict
    summaries: tp.List[tp.Tuple[tp.Optional[Module], str, tp.Any]]


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
            if value:  # drop if empty
                node[submodule] = value

        return node
    else:
        return ()


def set_tree(
    module: tp.Union[Module, tp.List, tp.Tuple, tp.Dict],
    values: tp.Union[tp.List, tp.Tuple, tp.Dict],
    dict_field: str,
):
    def f(module, values):
        dict_field_value = getattr(module, dict_field)
        for key, value in values.items():
            if key in dict_field_value:
                setattr(module, key, value)

    tree_apply(f, module, values)


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
