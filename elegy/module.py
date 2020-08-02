from contextlib import contextmanager
import threading
import typing as tp
from abc import ABCMeta, abstractmethod


import jax
import jax.numpy as jnp
import numpy as np

from elegy import utils

T = tp.TypeVar("T")
LOCAL = threading.local()
LOCAL.contexts = []


class ModuleMeta(ABCMeta):
    def __call__(cls: tp.Type[T], *args, **kwargs) -> T:
        module = cls.__new__(cls, *args, **kwargs)
        cls.__init__(module, *args, **kwargs)

        if (
            not hasattr(module, "name")
            or not hasattr(module, "_params")
            or not hasattr(module, "_states")
        ):
            raise ValueError(
                "Constructing an hk.Module without calling the super constructor "
                "is not supported."
            )


class Module(metaclass=ModuleMeta):
    """
    Basic Elegy Module. Its a thin wrapper around `hk.Module` that
    add custom functionalities related to Elegy.
    """

    name: str
    _params: tp.List[str]
    _states: tp.List[str]
    _next_states: tp.List[str]

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
        self._params = []
        self._states = []

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
        self, rng: tp.Optional[tp.Union[np.ndarray, int]] = None,
    ) -> "InitContext":
        """
        Initializes your function collecting parameters and states.
        """

        return InitContext(module=self, rng=rng)

    def apply(
        self,
        parameters: tp.Optional[tp.Dict] = None,
        states: tp.Optional[tp.Dict] = None,
        rng: tp.Optional[tp.Union[np.ndarray, int]] = None,
        training: bool = True,
        get_summaries: bool = False,
        return_context: bool = False,
    ) -> "ApplyContext":
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

        return ApplyContext(
            module=self,
            parameters=parameters,
            states=states,
            rng=rng,
            training=training,
            get_summaries=get_summaries,
            return_context=return_context,
        )

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
                setattr(self, name, initializer(shape, dtype))
                self._params.append(name)

            if context.get_summaries:
                base_name = context.unique_name[self]

                if (
                    base_name in context.parameters
                    and name not in context.parameters[base_name]
                ):
                    context.parameters[base_name][name] = getattr(self, name)
                else:
                    context.parameters[base_name] = {name: getattr(self, name)}

            param = getattr(self, name)

            assert param.shape == tuple(shape)
            assert param.dtype == dtype

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
        context: Context = LOCAL.contexts[-1]

        if LOCAL.contexts:
            if name not in self._states:
                setattr(self, name, initializer(shape, dtype))
                self._states.append(name)

                if context.get_summaries:
                    base_name = context.unique_name[self]

                    if (
                        base_name in context.states
                        and name not in context.states[base_name]
                    ):
                        context.states[base_name][name] = getattr(self, name)
                    else:
                        context.states[base_name] = {name: getattr(self, name)}

            param = getattr(self, name)

            assert param.shape == tuple(shape)
            assert param.dtype == dtype

            return param
        else:
            raise ValueError("Cannot execute `get_state` outside of a `elegy.context`")

    def set_state(self, name: str, value: tp.Any):

        if LOCAL.contexts:
            context: Context = LOCAL.contexts[-1]

            if not context.building:
                setattr(self, name, value)
        else:
            raise ValueError("Cannot execute `set_state` outside of a `elegy.context`")

    def add_loss(self, name: str, value: np.ndarray):
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

    def add_metric(self, name: str, value: np.ndarray):
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
            name = get_unique_name(context.metrics, name)
            context.metrics[name] = value
        else:
            raise ValueError(
                "Cannot execute `add_metric` outside of an `elegy.context`"
            )

    def add_summary(self, name: tp.Optional[str], value: np.ndarray):
        """
        A hook that lets you define a summary within a [`transform`][elegy.hooks.transform].

        ```python
        y = jax.nn.relu(x)
        elegy.add_summary("relu", "Relu", y)
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
            class_name = base_name.split("/")[-1]

            name = f"{base_name}/{name}" if name is not None else base_name
            name = get_unique_name(context.summaries, name)

            context.summaries.append((name, class_name, value))
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
        return set_tree(self, values, "_params")

    @property
    def states(self) -> tp.Dict:
        states = get_tree(self, "_states")
        assert isinstance(states, tp.Dict)
        return states

    @states.setter
    def states(self, values: tp.Dict):
        return set_tree(self, values, "_states")


# -------------------------------------------------------------
# context
# -------------------------------------------------------------


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
    training: tp.Optional[bool]
    rng: tp.Optional[np.ndarray]
    losses: tp.Dict
    metrics: tp.Dict
    summaries: tp.List[tp.Tuple[str, str, tp.Any]]
    names_context: tp.List[str]
    unique_name: tp.Dict[tp.Any, str]
    repeated_name_count: tp.Dict[str, int]
    parameters: tp.Dict[str, tp.Dict[str, tp.Any]]
    states: tp.Dict[str, tp.Dict[str, tp.Any]]


@contextmanager
def context(
    rng: tp.Union[np.ndarray, int, None] = None,
    training: tp.Optional[bool] = None,
    building: bool = False,
    get_summaries: bool = False,
) -> tp.Iterator[Context]:
    """
    """

    rng = jax.random.PRNGKey(rng) if rng is not None else None

    ctx = Context(
        building=building,
        get_summaries=get_summaries,
        training=training,
        rng=rng,
        losses={},
        metrics={},
        summaries=[],
        names_context=[],
        unique_name={},
        repeated_name_count={},
        parameters={},
        states={},
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


class InitContext(tp.NamedTuple):
    module: "Module"
    rng: tp.Optional[tp.Union[np.ndarray, int]]

    def __call__(self, *args, **kwargs) -> tp.Tuple[tp.Dict, tp.Dict]:
        with context(rng=self.rng, training=True, building=True, get_summaries=False):
            self.module(*args, **kwargs)

        return self.module.parameters, self.module.states


class ApplyContext(tp.NamedTuple):
    module: Module
    parameters: tp.Optional[tp.Dict]
    states: tp.Optional[tp.Dict]
    rng: tp.Optional[tp.Union[np.ndarray, int]]
    training: bool
    get_summaries: bool
    return_context: bool

    def __call__(self, *args, **kwargs) -> tp.Union[tp.Any, "ApplyOutput"]:

        if self.parameters:
            self.module.parameters = self.parameters

        if self.states:
            self.module.states = self.states

        with context(
            rng=self.rng,
            training=self.training,
            building=False,
            get_summaries=self.get_summaries,
        ) as ctx:
            outputs = self.module(*args, **kwargs)

        if self.return_context:
            return ApplyOutput(
                outputs=outputs,
                states=self.module.states,
                losses=ctx.losses,
                metrics=ctx.metrics,
                summaries=ctx.summaries,
                parameters_summaries=ctx.parameters,
                states_summaries=ctx.states,
            )
        else:
            return outputs


class ApplyOutput(tp.NamedTuple):
    outputs: tp.Any
    states: tp.Dict
    losses: tp.Dict
    metrics: tp.Dict
    summaries: tp.List[tp.Tuple[str, str, tp.Any]]
    parameters_summaries: tp.Dict[str, tp.Dict[str, tp.Any]]
    states_summaries: tp.Dict[str, tp.Dict[str, tp.Any]]


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
    losses: hk.State
    metrics: hk.State
    summaries: tp.List[tp.Tuple[str, str, tp.Any]]


# ------------------------------------------------------------------------
# utils
# ------------------------------------------------------------------------


def get_tree(
    module: tp.Union[Module, tp.List, tp.Tuple, tp.Dict], list_field: str
) -> tp.Union[tp.List, tp.Tuple, tp.Dict]:

    if isinstance(module, tp.List):
        return [get_tree(module, list_field) for module in module]
    elif isinstance(module, tp.Tuple):
        return tuple(get_tree(module, list_field) for module in module)
    elif isinstance(module, tp.Dict):
        return {key: get_tree(module, list_field) for key, module in module.items()}
    else:
        return dict(
            **{key: getattr(module, key) for key in getattr(module, list_field)},
            **{
                key: get_tree(value, list_field)
                for key, value in vars(module).items()
                if leaf_isinstance(value, Module)
            },
        )


def set_tree(
    module: tp.Union[Module, tp.List, tp.Tuple, tp.Dict],
    values: tp.Union[tp.List, tp.Tuple, tp.Dict],
    list_field: str,
):

    if isinstance(module, tp.List):
        assert isinstance(values, tp.List)

        for module, value in zip(module, values):
            set_tree(module, value, list_field)

    elif isinstance(module, tp.Tuple):
        assert isinstance(values, tp.Tuple)

        for module, value in zip(module, values):
            set_tree(module, value, list_field)

    elif isinstance(module, tp.Dict):
        assert isinstance(values, tp.Dict)

        for key, value in values.items():
            set_tree(module[key], value, list_field)

    else:
        assert isinstance(values, tp.Dict)

        for key, value in values.items():
            if key in getattr(module, list_field):
                setattr(module, key, value)
            else:
                set_tree(getattr(module, key), value, list_field)


def leaf_isinstance(obj: tp.Any, types) -> tp.Type:

    if isinstance(obj, (tp.List, tp.Tuple)) and obj:
        return leaf_isinstance(obj[0], types)
    elif isinstance(obj, tp.Dict) and obj:
        return leaf_isinstance(list(obj.values())[0], types)
    else:
        return isinstance(obj, types)


def get_unique_name(
    logs: tp.Union[
        tp.Set[str], tp.Dict[str, tp.Any], tp.List[tp.Tuple[str, str, tp.Any]]
    ],
    name: str,
):

    if isinstance(logs, dict):
        names = set(logs.keys())
    elif isinstance(logs, tp.List):
        names = {t[0] for t in logs}
    else:
        names = logs

    if name not in names:
        return name

    i = 1
    while f"{name}_{i}" in names:
        i += 1

    return f"{name}_{i}"


# -----------------------------------------------------------------
# PRNGSequence
# ----------------------------------------------------------------


class PRNGSequence:
    rng: np.ndarray

    def __init__(self, key: tp.Union[int, np.ndarray]):
        self.rng = jax.random.PRNGKey(key)

    def __next__(self):
        self.rng, rng_next = jax.random.split(self.rng)
        return rng_next
