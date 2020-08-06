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
LOCAL.init_contexts = []


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

    functools.wraps(module.call)(module)

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

    def get_submodules(self) -> tp.Dict[str, tp.Any]:
        return {name: getattr(self, name) for name in self._submodules}

    def init(
        self, rng: tp.Optional[tp.Union[np.ndarray, int]] = None
    ) -> "InitCallable":
        """
        Initializes your function collecting parameters and states.
        """

        @functools.wraps(self)
        def init_fn(*args, **kwargs):
            with context(rng=rng, building=True, get_summaries=False):
                self(*args, **kwargs)

            params = self.get_parameters()
            initial_states = self.get_states(initial=True)

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
    ) -> "ApplyCallable":
        """
        Applies your function injecting parameters and states.

        Arguments:
            parameters:
            states: 
            rng: 
            get_summaries: 
            args: 
            kwargs: 

        Returns:
            A [`TransformedState`][elegy.hooks.TransformedState] namedtuple consiting 
            of (outputs, states, losses, metrics, summaries).
        """

        @functools.wraps(self.call)
        def apply_fn(*args, **kwargs):

            current_parameters = self.get_parameters()
            current_states = self.get_states()

            assert current_parameters is not None
            assert current_states is not None

            if parameters is not None:
                self.set_parameters(parameters)

            if states is not None:
                self.set_states(states)

            with context(rng=rng, building=False, get_summaries=get_summaries,) as ctx:
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
        dtype: tp.Any = jnp.float32,
        initializer: tp.Callable[[tp.Sequence[int], tp.Any], np.ndarray] = None,
    ) -> np.ndarray:
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

            assert value.shape == tuple(shape)

            return value
        else:
            raise ValueError(
                "Cannot execute `get_parameter` outside of a `elegy.context`"
            )

    def add_state(
        self,
        name: str,
        shape: tp.Sequence[int],
        dtype: tp.Any = jnp.float32,
        initializer: tp.Callable[[tp.Sequence[int], tp.Any], tp.Any] = None,
    ) -> tp.Any:

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

            assert value.shape == tuple(shape)

            return value
        else:
            raise ValueError("Cannot execute `get_state` outside of a `elegy.context`")

    def update_state(self, name: str, value: tp.Any):

        if LOCAL.contexts:
            context: Context = LOCAL.contexts[-1]

            if name not in self._states:
                raise ValueError(f"State '{name}' not found.")
            setattr(self, name, value)
        else:
            raise ValueError("Cannot execute `set_state` outside of a `elegy.context`")

    def add_summary(self, name: tp.Optional[str], value: np.ndarray):
        """
        A hook that lets you define a summary within a [`transform`][elegy.hooks.transform].

        ```python
        y = jax.nn.relu(x)
        self.add_summary("relu", "Relu", y)
        ```

        The metrics will be aggregated by [`transform.apply`][elegy.hooks.transform.apply]
        and automatically handled by [`Model`][elegy.model.Model]. 

        Be default `add_summary` doesn't do anything, in order to enable the collection of
        summaries `get_summaries` must be set to `True`:

        ```python
        transformed_state = transform.apply(..., get_summaries=True, ...)
        ```

        [`Model.summary`][elegy.model.Model.summary] will render added summaries. 

        Arguments:
            name: The name of the loss. If a metric with the same
                `name` already exists a unique identifier will be generated.
            class_name:
            value: The value for the metric.
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
        params = get_tree(self, "_params")
        assert isinstance(params, tp.Dict)
        return params

    def set_parameters(self, values: types.Parameters):
        set_tree(self, values, "_params")

    def get_states(self, initial: bool = False) -> types.States:
        states = get_tree(
            self,
            "_states",
            key_fn=(lambda key: f"{key}__initial__") if initial else None,
        )
        assert isinstance(states, tp.Dict)
        return states

    def set_states(self, values: types.States):
        set_tree(self, values, "_states")

    def reset(self):
        def clear_module(module: Module):

            for name in module._params:
                delattr(module, name)

            for name in module._states:
                delattr(module, name)

            # module._params = set()
            # module._states = set()

        apply_tree(clear_module, self)

    def clear_initial_states(self):
        def clear_module(module: Module):
            for name in module._states_initial:
                delattr(module, name)

        apply_tree(clear_module, self)

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
    A hook that lets you define a loss within a [`transform`][elegy.hooks.transform].

    ```python
    w = hk.get_parameter("w", [3, 5], init=jnp.zeros)
    
    # L2 regularization penalty
    elegy.add_loss("l2_regularization", 0.01 * jnp.mean(w ** 2))
    ```

    The loss will be aggregated by [`transform.apply`][elegy.hooks.transform.apply]
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
    A hook that lets you define a metric within a [`transform`][elegy.hooks.transform].

    ```python
    y = jax.nn.relu(x)
    elegy.add_metric("activation_mean", jnp.mean(y))
    ```

    The metrics will be aggregated by [`transform.apply`][elegy.hooks.transform.apply]
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
    """
    A named tuple representing the outputs of [elegy.hooks.transform.apply][].

    Attributes:
        losses: The collected losses added by [`add_loss`][elegy.hooks.add_loss].
        metrics: The collected metrics added by [`add_metric`][elegy.hooks.add_metric].
        summaries: A list of `(name, class_name, value)` tuples
            added by [`add_summary`][elegy.hooks.add_summary].
    """

    building: bool
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
) -> tp.Iterator[Context]:
    """
    """

    rng_sequence = PRNGSequence(rng) if rng is not None else None

    ctx = Context(
        building=building,
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


class TransformedState(tp.NamedTuple):
    """
    A named tuple representing the outputs of [elegy.hooks.transform.apply][].

    Attributes:
        outputs: The output of the transformed function.
        losses: The collected losses added by [`add_loss`][elegy.hooks.add_loss].
        metrics: The collected metrics added by [`add_metric`][elegy.hooks.add_metric].
        summaries: A list of `(name, class_name, value)` tuples
            added by [`add_summary`][elegy.hooks.add_summary].
    """

    outputs: tp.Any
    losses: tp.Dict
    metrics: tp.Dict
    summaries: tp.List[tp.Tuple[tp.Optional[Module], str, tp.Any]]


# ------------------------------------------------------------------------
# utils
# ------------------------------------------------------------------------


def get_tree(
    module: tp.Union[Module, tp.List, tp.Tuple, tp.Dict],
    dict_field: str,
    key_fn: tp.Optional[tp.Callable[[str], str]] = None,
) -> tp.Union[tp.List, tp.Tuple, tp.Dict]:

    if isinstance(module, tp.List):
        return [get_tree(module, dict_field, key_fn) for module in module]
    elif isinstance(module, tp.Tuple):
        return tuple(get_tree(module, dict_field, key_fn) for module in module)
    elif isinstance(module, tp.Dict):
        return {
            key: get_tree(module, dict_field, key_fn) for key, module in module.items()
        }
    elif isinstance(module, Module):

        node = {
            key: getattr(module, key_fn(key))
            if key_fn is not None
            else getattr(module, key)
            for key in getattr(module, dict_field)
            if hasattr(module, key)
        }

        for submodule in module._submodules:
            value = get_tree(getattr(module, submodule), dict_field, key_fn)
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

    if isinstance(module, tp.List):
        assert isinstance(values, tp.List)

        for module, value in zip(module, values):
            set_tree(module, value, dict_field)

    elif isinstance(module, tp.Tuple):
        assert isinstance(values, tp.Tuple)

        for module, value in zip(module, values):
            set_tree(module, value, dict_field)

    elif isinstance(module, tp.Dict):
        assert isinstance(values, tp.Dict)

        for key, value in values.items():
            set_tree(module[key], value, dict_field)

    elif isinstance(module, Module):
        assert isinstance(values, tp.Dict)

        dict_field_value = getattr(module, dict_field)

        for key, value in values.items():
            if key in module._submodules:
                set_tree(getattr(module, key), value, dict_field)
            elif key in dict_field_value:
                setattr(module, key, value)


def apply_tree(f: tp.Callable, module: tp.Union[Module, tp.List, tp.Tuple, tp.Dict]):

    if isinstance(module, tp.List):

        for module in module:
            apply_tree(f, module)

    elif isinstance(module, tp.Tuple):

        for module in module:
            apply_tree(f, module)

    elif isinstance(module, tp.Dict):

        for key, value in module.items():
            apply_tree(f, module[key])

    elif isinstance(module, Module):
        f(module)

        for key in module._submodules:
            apply_tree(f, getattr(module, key))


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
        def __init__(self):
            super().__init__(name=utils.lower_snake_case(f.__name__))
            self.call = f

        def call(self, *args, **kwargs):
            ...

    MyModule.__name__ = f.__name__

    return MyModule

