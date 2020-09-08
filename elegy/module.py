import functools
import threading
import typing as tp
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

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


class LocalContext(utils.Protocol):
    rng: RNG
    mode: Mode
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
    mode: Mode
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
    mode=Mode.train,
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
    rng: tp.Optional[RNG] = None,
    mode: tp.Optional[Mode] = None,
    hooks: bool = False,
) -> tp.Iterator[LocalContext]:

    prev_rng = LOCAL.rng
    prev_mode = LOCAL.mode
    prev_losses = LOCAL.losses
    prev_metrics = LOCAL.metrics
    prev_summaries = LOCAL.summaries
    prev_names = LOCAL.names
    prev_level_names = LOCAL.level_names

    LOCAL.rng = rng if rng is not None else LOCAL.rng
    LOCAL.mode = mode if mode is not None else LOCAL.mode
    LOCAL.losses = {} if hooks else LOCAL.losses
    LOCAL.metrics = {} if hooks else LOCAL.metrics
    LOCAL.summaries = [] if hooks else LOCAL.summaries
    LOCAL.names = [] if hooks else LOCAL.names
    LOCAL.level_names = [[]] if hooks else LOCAL.level_names

    try:
        yield _LocalContext(**vars(LOCAL))
    finally:
        LOCAL.rng = prev_rng
        LOCAL.mode = prev_mode
        LOCAL.losses = prev_losses
        LOCAL.metrics = prev_metrics
        LOCAL.summaries = prev_summaries
        LOCAL.names = prev_names
        LOCAL.level_names = prev_level_names


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
    # utils.wraps(module.call)(module.init)

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

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        utils.wraps(cls.call)(cls.init)

    @abstractmethod
    def call(self, *args, **kwargs):
        ...

    def jit(self, *args, **kwargs):

        outputs, parameters, rng = self._call_jit(
            get_mode(), get_rng(), self.get_parameters(), args, kwargs
        )

        self.set_parameters(parameters)
        set_rng(rng)

        return outputs

    @functools.partial(jax.jit, static_argnums=(0, 1))
    def _call_jit(
        self,
        mode: Mode,
        rng: RNG,
        parameters: tp.Dict,
        args,
        kwargs,
    ):
        self.set_parameters(parameters)

        with context(rng=rng, mode=mode):
            outputs = self(*args, **kwargs)

        return outputs, self.get_parameters(), rng

    def init_jit(self, *args, **kwargs):
        parameters, rng = self._init_jit(get_mode(), get_rng(), args, kwargs)

        self.set_parameters(parameters)
        set_rng(rng)

    @functools.partial(jax.jit, static_argnums=(0, 1))
    def _init_jit(
        self,
        mode: Mode,
        rng: RNG,
        args,
        kwargs,
    ):

        with context(rng=rng, mode=mode):
            self.init(*args, **kwargs)

        return self.get_parameters(), rng

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

    def get_parameters(
        self,
        trainable: tp.Optional[bool] = None,
        non_trainable: tp.Optional[bool] = None,
    ) -> types.Parameters:
        """
        Recursively collects a dictionary with the parameters of this module
        and all submodules within it.

        Returns:
        """
        if trainable is None and non_trainable is None:
            trainable = True
            non_trainable = True

        if not trainable and not non_trainable:
            raise ValueError(
                f"Must specify either trainable or non_trainable, "
                "got trainable = {trainable}, non_trainable = {non_trainable}"
            )

        params = module_tree_map(
            lambda module: {
                key: getattr(module, key)
                for key in (
                    (module._params if trainable else [])
                    + (module._states if non_trainable else [])
                )
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
        set_tree(self, values, ["_params", "_states"])

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
                x.size for x in jax.tree_leaves(self.get_parameters(non_trainable=True))
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


def get_rng() -> RNG:
    return LOCAL.rng


def set_rng(rng: RNG) -> None:
    LOCAL.rng = rng


def next_rng_key() -> jnp.ndarray:
    return LOCAL.rng()


def module_initializing() -> bool:
    return LOCAL.initializing


def is_training() -> bool:
    return get_mode() == Mode.train


def get_mode() -> Mode:
    return LOCAL.mode


def set_mode(mode: Mode) -> None:
    LOCAL.mode = mode


def get_losses() -> tp.Optional[tp.Dict[str, tp.Any]]:
    return LOCAL.losses


def get_metrics() -> tp.Optional[tp.Dict[str, tp.Any]]:
    return LOCAL.metrics


def get_summaries() -> tp.Optional[
    tp.List[tp.Tuple[tp.Optional["Module"], str, np.ndarray]]
]:
    return LOCAL.summaries


def base_name() -> str:
    assert LOCAL.names
    return "/".join(LOCAL.names)


@contextmanager
def rng_context(rng: RNG):
    current_rng = LOCAL.rng
    LOCAL.rng = rng

    try:
        yield
    finally:
        LOCAL.rng = current_rng


@contextmanager
def mode_context(mode: Mode):
    current_mode = LOCAL.mode
    LOCAL.mode = mode

    try:
        yield
    finally:
        LOCAL.mode = current_mode


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
    field_names: tp.List[str],
):
    def f(module, values):
        names = sum((getattr(module, field_name) for field_name in field_names), [])
        for key, value in values.items():
            if key in names:
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
