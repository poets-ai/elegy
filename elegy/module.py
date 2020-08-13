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
                        parent_module, parent_module._dynamic_submodules[index],
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
    _params: tp.Set[str]
    _states: tp.Set[str]
    _states_initial: tp.Set[str]
    _submodules: tp.Set[str]
    _dynamic_submodules: tp.List[str]
    _ignore: tp.Set[str]

    def __init__(self, name: tp.Optional[str] = None):
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
        self._params = set()
        self._states = set()
        self._states_initial = set()
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

            self.add_summary(None, outputs)

            return outputs

    @abstractmethod
    def call(self, *args, **kwargs):
        ...

    @property
    def submodules(self) -> tp.Dict[str, tp.Any]:
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
                rng=rng, building=False, training=training, get_summaries=get_summaries,
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

    def add_parameter(
        self,
        name: str,
        shape: tp.Sequence[int],
        dtype: tp.Any,
        initializer: tp.Callable[[tp.Sequence[int], tp.Any], np.ndarray],
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
        if LOCAL.contexts:
            context: Context = LOCAL.contexts[-1]

            if not hasattr(self, name):
                if not context.building:
                    raise ValueError(
                        f"Trying to initialize '{name}' outside of `init`."
                    )

                self._params.add(name)
                setattr(self, name, initializer(shape, dtype))

            elif name not in self._params:
                raise ValueError(
                    f"Class already contained a property named '{name}', "
                    "please use a unique name for the parameter."
                )

            value = getattr(self, name)

            return value
        else:
            raise ValueError(
                "Cannot execute `get_parameter` outside of a `elegy.context`"
            )

    def add_state(
        self,
        name: str,
        shape: tp.Sequence[int],
        dtype: tp.Any,
        initializer: tp.Callable[[tp.Sequence[int], tp.Any], tp.Any],
    ) -> tp.Any:
        """
        A hook that lets you add a state to the current module. The state will only be created once
        during `init` and will reused afterwards.

        Arguments:
            name: The name of the state. It must be unique and no other field/property/method
                of the instance can have that name.
            shape: The shape of the state.
            dtype: The type of the state.
            initializer: A callable that takes in a shape and dtype and returns the initial value.

        Returns:
            The value of the state.
        """

        if LOCAL.contexts:
            context: Context = LOCAL.contexts[-1]

            if not hasattr(self, name):
                if not context.building:
                    raise ValueError(
                        f"Trying to initialize '{name}' outside of `init`."
                    )

                initial_name = f"{name}__initial__"

                self._states.add(name)
                self._states_initial.add(initial_name)

                initial_value = initializer(shape, dtype)

                setattr(self, name, initial_value)
                setattr(self, initial_name, initial_value)

            elif name not in self._states:
                raise ValueError(
                    f"Class already contained a property named '{name}', "
                    "please use a unique name for the state."
                )

            value = getattr(self, name)

            return value
        else:
            raise ValueError("Cannot execute `get_state` outside of a `elegy.context`")

    def update_state(self, name: str, value: tp.Any):
        """
        A hook that lets you ypdate a state of the current module, if the state does not 
        exist it will be created.

        Arguments:
            name: The name of the state. It must be unique and no other field/property/method
                of the instance can have that name.
            value: The updated value of the state.

        Returns:
            The value of the state.
        """
        if LOCAL.contexts:
            context: Context = LOCAL.contexts[-1]

            if name not in self._states:
                if not context.building:
                    raise ValueError(
                        f"Trying to initialize '{name}' outside of `init`."
                    )

                initial_name = f"{name}__initial__"

                self._states.add(name)
                self._states_initial.add(initial_name)

                setattr(self, name, value)
                setattr(self, initial_name, value)
            else:
                setattr(self, name, value)
        else:
            raise ValueError("Cannot execute `set_state` outside of a `elegy.context`")

    def add_summary(self, name: tp.Optional[str], value: np.ndarray):
        """
        A hook that lets you define a summary in the current module. Its primary
        use is to keep track of certain values as they flow through the network
        so `Model.summary()` can show a representation of architecture.

        ```python
        def call(self, x):
            ...
            y = jax.nn.relu(x)
            self.add_summary("relu", y)
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

        if LOCAL.contexts:
            context: Context = LOCAL.contexts[-1]

            if not context.get_summaries:
                return

            # name = level_names[self]
            base_name = "/".join(context.path_names_c)

            base_name = f"{base_name}/{name}" if name is not None else base_name
            base_name = get_unique_name(context.summaries, base_name)
            module = self if name is None else None  # pass module only if name is None

            context.summaries.append((module, base_name, value))
        else:
            raise ValueError(
                "Cannot execute `add_summary` outside of an `elegy.context`"
            )

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
                key: getattr(module, key_initial) if _initial else getattr(module, key)
                for key, key_initial in zip(
                    getattr(module, "_states"), getattr(module, "_states_initial")
                )
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
            for name in module._states_initial:
                delattr(module, name)

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


def add_loss(name: str, value: np.ndarray):
    """
    A hook that lets you define a loss within a [`module`][elegy.module.Module].

    ```python
    w = hk.get_parameter("w", [3, 5], init=jnp.zeros)
    
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
    name += "_loss"
    if LOCAL.contexts:
        context: Context = LOCAL.contexts[-1]

        if name in context.losses:
            context.losses[name] += value
        else:
            context.losses[name] = value
    else:
        raise ValueError("Cannot execute `add_loss` outside of an `elegy.context`")


def add_metric(name: str, value: np.ndarray):
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
    if LOCAL.contexts:
        context: Context = LOCAL.contexts[-1]

        base_name = "/".join(context.path_names_c)
        name = f"{base_name}/{name}"
        name = get_unique_name(context.metrics, name)
        context.metrics[name] = value
    else:
        raise ValueError("Cannot execute `add_metric` outside of an `elegy.context`")


def next_rng_key() -> PRNGKey:
    """
    Returns a unique JAX RNG key split from the current global key.

    ```python
    key = hk.next_rng_key()
    x = jax.random.uniform(key, [])
    ```

    Returns:
        A unique (within a transformed function) JAX rng key that can be used with
        APIs such as ``jax.random.uniform``.
    """
    if LOCAL.contexts:
        context: Context = LOCAL.contexts[-1]
        if context.rng_sequence is not None:
            context: Context = LOCAL.contexts[-1]
            return next(context.rng_sequence)
        else:
            raise ValueError(
                "Cannot execute `rng` not set in context, check init or apply."
            )
    else:
        raise ValueError("Cannot execute `next_rng_key` outside of an `elegy.context`")


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
    """
    """

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

    current_module = context.module_c

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
