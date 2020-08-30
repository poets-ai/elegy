import functools
import re
import threading
import typing as tp
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager

# -------------------------------------------------------------
# context
# -------------------------------------------------------------
from dataclasses import dataclass

# -----------------------------------------------------------------
# PRNGSequence
# ----------------------------------------------------------------
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from elegy import types, utils
from elegy.types import PRNGKey

__all__ = [
    # "ApplyCallable",
    # "ApplyContext",
    # "Context",
    # "InitCallable",
    "Module",
    "to_module",
]

T = tp.TypeVar("T")
LOCAL = threading.local()
LOCAL.contexts = []


def construct_module(cls, *args, **kwargs) -> "Module":
    module: Module = cls.__new__(cls, *args, **kwargs)
    with initialization_context(module):
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
            module._submodules.add(key)

    utils.wraps(module.call)(module)

    return module


class ModuleMeta(ABCMeta):
    def __call__(cls: tp.Type[T], *args, **kwargs) -> "Module":
        module: Module
        context: Context

        if LOCAL.contexts:
            context = LOCAL.contexts[-1]
            parent_module = context.module_c[-1] if context.module_c else None

            # Set unique on parent when using inside `call`
            if parent_module is not None and context.inside_call_c[-1]:
                index = context.index_c[-1]
                assert parent_module is not None

                if len(parent_module._dynamic_submodules) > index:
                    module = getattr(
                        parent_module,
                        parent_module._dynamic_submodules[index],
                    )
                else:
                    if not context.building:
                        raise ValueError(
                            f"Trying to create module of type'{cls.__name__}' outside of `init`."
                        )

                    module = construct_module(cls, *args, **kwargs)

                    name = get_unique_name(parent_module, module.name)
                    setattr(parent_module, name, module)
                    parent_module._submodules.add(name)
                    parent_module._dynamic_submodules.append(name)

                assert module is not None
                context.index_c[-1] += 1

                return module

        return construct_module(cls, *args, **kwargs)


class Module(metaclass=ModuleMeta):
    """
    Basic Elegy Module. Its a thin wrapper around `hk.Module` that
    add custom functionalities related to Elegy.
    """

    name: str
    dtype: np.dtype
    _params: tp.Set[str]
    _states: tp.Set[str]
    _states_initial: tp.Set[str]
    _submodules: tp.Set[str]
    _dynamic_submodules: tp.List[str]
    _ignore: tp.Set[str]

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
        self._params = set()
        self._states = set()
        self._submodules = set()
        self._dynamic_submodules = []
        self._ignore = set()

    def __call__(self, *args, **kwargs) -> tp.Any:
        """
        Forwards all input arguments to the Module's `call` method and calls
        `elegy.add_summary` on the outputs.
        """
        with call_context(self):
            outputs = self.call(*args, **kwargs)

            add_summary(None, outputs)

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

    def init(
        self, rng: tp.Optional[tp.Union[np.ndarray, int]] = None
    ) -> "InitCallable":
        """
        Initializes the module.

        Arguments:
            rng:

        Returns:

        """

        @utils.wraps(self)
        def init_fn(*args, **kwargs):
            with context(rng=rng, building=True, get_summaries=False):
                self(*args, **kwargs)

            params = self.get_parameters()
            initial_states = self.get_states(_initial=True)

            self.clear_initial_states()
            self.set_states(initial_states)

            return params, initial_states

        return init_fn

    def apply(
        self,
        parameters: tp.Optional[tp.Dict] = None,
        states: tp.Optional[tp.Dict] = None,
        rng: tp.Optional[tp.Union[np.ndarray, int]] = None,
        get_summaries: bool = False,
        training: bool = True,
    ) -> "ApplyCallable":
        """
        Applies your function injecting some context arguments.

        Arguments:
            parameters:
            states:
            rng:
            get_summaries:

        Returns:
            A function with the same signature as `call` that will
            execute the computation given the context arguments
            passed to `apply`.
        """

        @utils.wraps(self.call)
        def apply_fn(*args, **kwargs):

            current_parameters = self.get_parameters()
            current_states = self.get_states()

            assert current_parameters is not None
            assert current_states is not None

            if parameters is not None:
                self.set_parameters(parameters)

            if states is not None:
                self.set_states(states)

            with context(
                rng=rng,
                building=False,
                training=training,
                get_summaries=get_summaries,
            ) as ctx:
                outputs = self(*args, **kwargs)

            output_parameters = self.get_parameters()
            output_states = self.get_states()

            if parameters is not None:
                self.set_parameters(current_parameters)

            if states is not None:
                self.set_states(current_states)

            return (
                outputs,
                ApplyContext(
                    parameters=output_parameters,
                    states=output_states,
                    losses=ctx.losses,
                    metrics=ctx.metrics,
                    summaries=ctx.summaries,
                ),
            )

        return apply_fn

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

# this funtion is patched in hooks
def add_summary(name: tp.Optional[str], value: np.ndarray):
    raise NotImplementedError()


@dataclass
class Context:
    building: bool
    training: bool
    get_summaries: bool
    rng_sequence: tp.Optional["PRNGSequence"]
    losses: tp.Dict
    metrics: tp.Dict
    summaries: tp.List[tp.Tuple[tp.Optional[Module], str, tp.Any]]
    path_names_c: tp.List[str]
    level_names_c: tp.List[tp.Dict[Module, str]]
    inside_call_c: tp.List[bool]
    module_c: tp.List[Module]
    index_c: tp.List[int]


@contextmanager
def context(
    rng: tp.Union[np.ndarray, int, None] = None,
    building: bool = False,
    get_summaries: bool = False,
    training: bool = True,
) -> tp.Iterator[Context]:
    """"""

    rng_sequence = PRNGSequence(rng) if rng is not None else None

    ctx = Context(
        building=building,
        training=training,
        get_summaries=get_summaries,
        rng_sequence=rng_sequence,
        losses={},
        metrics={},
        summaries=[],
        path_names_c=[],
        level_names_c=[],
        inside_call_c=[],
        module_c=[],
        index_c=[],
    )

    LOCAL.contexts.append(ctx)

    if rng is not None:
        rng = hk.PRNGSequence(rng)

    try:
        yield ctx
    finally:
        LOCAL.contexts.pop()


@contextmanager
def call_context(module: Module):

    if LOCAL.contexts:
        context: Context = LOCAL.contexts[-1]

        if context.level_names_c:
            level_names = context.level_names_c[-1]

            if module not in level_names:
                name = get_unique_name(set(level_names.values()), module.name)
                level_names[module] = name
            else:
                name = level_names[module]

        else:
            name = module.name

        context.path_names_c.append(name)
        context.inside_call_c.append(True)
        context.module_c.append(module)
        context.index_c.append(0)
        context.level_names_c.append({})

        try:
            yield
        finally:
            context.path_names_c.pop()
            context.inside_call_c.pop()
            context.module_c.pop()
            context.index_c.pop()
            context.level_names_c.pop()
    else:
        raise ValueError("Cannot execute `call` outside of a `elegy.context`")


@contextmanager
def initialization_context(module: Module):

    context: Context

    if LOCAL.contexts:
        pop_context = False
        context = LOCAL.contexts[-1]
    else:
        pop_context = True
        context = Context(
            building=False,
            training=True,
            get_summaries=False,
            rng_sequence=None,
            losses={},
            metrics={},
            summaries=[],
            path_names_c=[],
            level_names_c=[],
            inside_call_c=[],
            module_c=[],
            index_c=[],
        )
        LOCAL.contexts.append(context)

    context.inside_call_c.append(False)
    context.module_c.append(module)

    try:
        yield
    finally:
        if pop_context:
            LOCAL.contexts.pop()
        else:
            context.inside_call_c.pop()
            context.module_c.pop()


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


def tree_exec(f: tp.Callable, module: tp.Union[Module, tp.List, tp.Tuple, tp.Dict]):

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
    logs: tp.Union[
        tp.Set[str],
        tp.Dict[str, tp.Any],
        tp.List[tp.Tuple[tp.Optional[Module], str, tp.Any]],
        Module,
    ],
    name: str,
):

    if isinstance(logs, dict):
        names = set(logs.keys())
    elif isinstance(logs, tp.List):
        names = {t[1] for t in logs}
    elif isinstance(logs, Module):
        names = set(vars(logs).keys())
    else:
        names = logs

    if name not in names:
        return name

    i = 1
    while f"{name}_{i}" in names:
        i += 1

    return f"{name}_{i}"


hk.PRNGSequence


class PRNGSequence(tp.Iterator[PRNGKey]):
    key: np.ndarray

    def __init__(self, key: tp.Union[int, np.ndarray]):
        self.key = (
            jax.random.PRNGKey(key) if isinstance(key, int) or key.shape == () else key
        )

    def __next__(self) -> np.ndarray:
        self.key, rng_next = tuple(jax.random.split(self.key, 2))
        return rng_next

    next = __next__


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
