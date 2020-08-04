import functools
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


class ModuleMeta(ABCMeta):
    def __call__(cls: tp.Type[T], *args, **kwargs) -> T:
        module: Module = cls.__new__(cls, *args, **kwargs)
        cls.__init__(module, *args, **kwargs)
        functools.wraps(module.call)(module)

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
            if leaf_isinstance(value, Module):
                module._submodules.add(key)

        return module


class Module(metaclass=ModuleMeta):
    """
    Basic Elegy Module. Its a thin wrapper around `hk.Module` that
    add custom functionalities related to Elegy.
    """

    name: str
    _params: tp.Dict[str, tp.Any]
    _states: tp.Dict[str, tp.Any]
    _submodules: tp.Set[str]

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
        self._params = {}
        self._states = {}
        self._submodules = set()

    def __call__(self, *args, **kwargs) -> tp.Any:
        """
        Forwards all input arguments to the Module's `call` method and calls
        `elegy.add_summary` on the outputs.
        """
        with names_context(self):
            outputs = self.call(*args, **kwargs)

            self.add_summary(None, outputs)

            return outputs

    @abstractmethod
    def call(self, *args, **kwargs):
        ...

    def init(
        self, rng: tp.Optional[tp.Union[np.ndarray, int]] = None
    ) -> "InitCallable":
        """
        Initializes your function collecting parameters and states.
        """

        @functools.wraps(self.call)
        def init_fn(*args, **kwargs):
            with context(rng=rng, building=True, get_summaries=False):
                self(*args, **kwargs)

            return self.parameters, self.states

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

            if parameters:
                self.parameters = parameters

            if states:
                self.states = states

            with context(rng=rng, building=False, get_summaries=get_summaries,) as ctx:
                outputs = self(*args, **kwargs)

            return (
                outputs,
                ApplyContext(
                    states=self.states,
                    losses=ctx.losses,
                    metrics=ctx.metrics,
                    summaries=ctx.summaries,
                ),
            )

        return apply_fn

    def get_parameter(
        self,
        name: str,
        shape: tp.Sequence[int],
        dtype: tp.Any = jnp.float32,
        initializer: tp.Callable[[tp.Sequence[int], tp.Any], np.ndarray] = None,
    ) -> np.ndarray:
        if LOCAL.contexts:
            context: Context = LOCAL.contexts[-1]

            if name not in self._params:
                if not context.building:
                    raise ValueError(
                        f"Trying to initialize '{name}' outside of `init`."
                    )

                if name in self._submodules:
                    raise ValueError(
                        f"Cannot use name '{name}' for parameter since a submodule "
                        "with the same name already exists."
                    )

                self._params[name] = initializer(shape, dtype)

            param = self._params[name]

            assert param.shape == tuple(shape)

            return param
        else:
            raise ValueError(
                "Cannot execute `get_parameter` outside of a `elegy.context`"
            )

    def get_state(
        self,
        name: str,
        shape: tp.Sequence[int],
        dtype: tp.Any = jnp.float32,
        initializer: tp.Callable[[tp.Sequence[int], tp.Any], tp.Any] = None,
    ) -> tp.Any:

        if LOCAL.contexts:
            context: Context = LOCAL.contexts[-1]

            if name not in self._states:
                if not context.building:
                    raise ValueError(
                        f"Trying to initialize '{name}' outside of `init`."
                    )

                if name in self._submodules:
                    raise ValueError(
                        f"Cannot use name '{name}' for state since a submodule "
                        "with the same name already exists."
                    )

                self._states[name] = initializer(shape, dtype)

            param = self._states[name]

            assert param.shape == tuple(shape)

            return param
        else:
            raise ValueError("Cannot execute `get_state` outside of a `elegy.context`")

    def set_state(self, name: str, value: tp.Any):

        if LOCAL.contexts:
            context: Context = LOCAL.contexts[-1]

            if not context.building:
                self._states[name] = value
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

            base_name = context.unique_name[self]

            base_name = f"{base_name}/{name}" if name is not None else base_name
            base_name = get_unique_name(context.summaries, base_name)
            module = self if name is None else None  # pass module only if name is None

            context.summaries.append((module, base_name, value))
        else:
            raise ValueError(
                "Cannot execute `add_summary` outside of an `elegy.context`"
            )

    @property
    def parameters(self) -> tp.Dict:
        parameters = get_tree(self, "_params")
        assert isinstance(parameters, tp.Dict)
        return parameters

    @parameters.setter
    def parameters(self, values: tp.Dict):
        set_tree(self, values, "_params")

    def clear_parameters(self):
        clear_tree(self, "_params")

    @property
    def states(self) -> tp.Dict:
        states = get_tree(self, "_states")
        assert isinstance(states, tp.Dict)
        return states

    @states.setter
    def states(self, values: tp.Dict):
        set_tree(self, values, "_states")

    def clear_states(self):
        clear_tree(self, "_states")

    def parameters_size(self, include_submodules: bool = True):
        if include_submodules:
            return sum(x.size for x in jax.tree_leaves(self.parameters))
        else:
            return sum(x.size for x in jax.tree_leaves(self._params))

    def states_size(self, include_submodules: bool = True):
        if include_submodules:
            return sum(x.size for x in jax.tree_leaves(self.states))
        else:
            return sum(x.size for x in jax.tree_leaves(self._states))

    def parameters_bytes(self, include_submodules: bool = True):
        if include_submodules:
            return sum(
                x.size * x.dtype.itemsize for x in jax.tree_leaves(self.parameters)
            )
        else:
            return sum(x.size * x.dtype.itemsize for x in jax.tree_leaves(self._params))

    def states_bytes(self, include_submodules: bool = True):
        if include_submodules:
            return sum(x.size * x.dtype.itemsize for x in jax.tree_leaves(self.states))
        else:
            return sum(x.size * x.dtype.itemsize for x in jax.tree_leaves(self._states))


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
        base_name = "/".join(context.names_context)
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


class Context(tp.NamedTuple):
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
    names_context: tp.List[str]
    unique_name: tp.Dict[tp.Any, str]
    repeated_name_count: tp.Dict[str, int]


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
        names_context=[],
        unique_name={},
        repeated_name_count={},
    )

    LOCAL.contexts.append(ctx)

    if rng is not None:
        rng = hk.PRNGSequence(rng)

    try:
        yield ctx
    finally:
        LOCAL.contexts.pop()


@contextmanager
def names_context(module: Module):

    if LOCAL.contexts:
        context: Context = LOCAL.contexts[-1]

        class_name = module.__class__.__name__

        # get unique name
        if module in context.unique_name:
            base_name = context.unique_name[module]
        else:
            class_name = utils.lower_snake_case(class_name)
            base_name = (
                "/".join(context.names_context) + f"/{class_name}"
                if context.names_context
                else class_name
            )

            if base_name in context.repeated_name_count:
                context.repeated_name_count[base_name] += 1
                base_name += f"_{context.repeated_name_count[base_name] - 1}"
            else:
                context.repeated_name_count[base_name] = 1

            context.unique_name[module] = base_name

        # get class name
        class_name = base_name.split("/")[-1]
        context.names_context.append(class_name)

        try:
            yield
        finally:
            context.names_context.pop()
    else:
        raise ValueError("")


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
    states: tp.Dict
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
    module: tp.Union[Module, tp.List, tp.Tuple, tp.Dict], dict_field: str
) -> tp.Union[tp.List, tp.Tuple, tp.Dict]:

    if isinstance(module, tp.List):
        return [get_tree(module, dict_field) for module in module]
    elif isinstance(module, tp.Tuple):
        return tuple(get_tree(module, dict_field) for module in module)
    elif isinstance(module, tp.Dict):
        return {key: get_tree(module, dict_field) for key, module in module.items()}
    elif isinstance(module, Module):
        node = getattr(module, dict_field).copy()
        for submodule in module._submodules:
            value = get_tree(getattr(module, submodule), dict_field)
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

        for key in dict_field_value:
            dict_field_value[key] = values[key]

        for key in module._submodules:
            if key in values:
                set_tree(getattr(module, key), values[key], dict_field)


def clear_tree(module: tp.Union[Module, tp.List, tp.Tuple, tp.Dict], dict_field: str):

    if isinstance(module, tp.List):

        for module in module:
            clear_tree(module, dict_field)

    elif isinstance(module, tp.Tuple):

        for module in module:
            clear_tree(module, dict_field)

    elif isinstance(module, tp.Dict):

        for key, value in module.items():
            clear_tree(module[key], dict_field)

    else:
        setattr(module, dict_field, {})

        for key in module._submodules:
            clear_tree(getattr(module, key), dict_field)


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
    ],
    name: str,
):

    if isinstance(logs, dict):
        names = set(logs.keys())
    elif isinstance(logs, tp.List):
        names = {t[1] for t in logs}
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
    rng: np.ndarray

    def __init__(self, key: tp.Union[int, np.ndarray]):
        self.rng = (
            jax.random.PRNGKey(key) if isinstance(key, int) or key.shape == () else key
        )

    def __next__(self) -> np.ndarray:
        self.rng, rng_next = tuple(jax.random.split(self.rng, 2))
        return rng_next

    next = __next__
