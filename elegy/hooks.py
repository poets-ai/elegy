# Some portiong of this code are adapted from Haiku:
# https://github.com/deepmind/dm-haiku/blob/master/haiku/_src/transform.py#L228#L300

import threading
import typing as tp
from contextlib import contextmanager

import haiku as hk
import numpy as np
from haiku._src import base
from haiku._src import transform as src_transform


T = tp.TypeVar("T")
LOCAL = threading.local()
LOCAL.contexts = []


class Context(tp.NamedTuple):
    get_summaries: bool
    losses: tp.Dict[str, np.ndarray]
    metrics: tp.Dict[str, np.ndarray]
    summaries: tp.List[tp.Tuple[str, str, tp.Any]]

    @classmethod
    def create(cls, get_summaries: bool = False):
        return cls(get_summaries=get_summaries, losses={}, metrics={}, summaries=[])


@contextmanager
def elegy_context(get_summaries: bool = False):

    context = Context.create(get_summaries=get_summaries)
    LOCAL.contexts.append(context)
    try:
        yield context
    finally:
        LOCAL.contexts.pop()


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
        raise ValueError("Cannot execute `add_loss` outside of an `elegy.transform`")


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

        name = f"{base.current_bundle_name()}/{name}"
        name = get_unique_name(context.metrics, name)

        context.metrics[name] = value
    else:
        raise ValueError("Cannot execute `add_metric` outside of an `elegy.transform`")


def add_summary(name: tp.Optional[str], class_name, value: np.ndarray):
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

        base_names = base.current_bundle_name()
        name = f"{base_names}/{name}" if name is not None else base_names
        name = get_unique_name(context.summaries, name)

        context.summaries.append((name, class_name, value))
    else:
        raise ValueError("Cannot execute `add_summary` outside of an `elegy.transform`")


def get_unique_name(
    logs: tp.Union[tp.Dict[str, tp.Any], tp.List[tp.Tuple[str, str, tp.Any]]], name: str
):

    names: tp.Set[str] = set(logs.values()) if isinstance(logs, dict) else {
        t[0] for t in logs
    }

    if name not in names:
        return name

    i = 1
    while f"{name}_{i}" in names:
        i += 1

    return f"{name}_{i}"


class TransformedState(tp.NamedTuple):
    """
    A named tuple representing the outputs of [elegy.hooks.transform.apply][].

    Attributes:
        outputs: The output of the transformed function.
        state: The states parameters.
        losses: The collected losses added by [`add_loss`][elegy.hooks.add_loss].
        metrics: The collected metrics added by [`add_metric`][elegy.hooks.add_metric].
        summaries: A list of `(name, class_name, value)` tuples
            added by [`add_summary`][elegy.hooks.add_summary].
    """

    outputs: tp.Any
    state: hk.State
    losses: hk.State
    metrics: hk.State
    summaries: tp.List[tp.Tuple[str, str, tp.Any]]


class transform(tp.NamedTuple):
    """
    Transforms a function using Elegy modules into pure functions.

    `transform` is a stronger version of 
    [hk.transform_with_state](https://dm-haiku.readthedocs.io/en/latest/api.html?highlight=transform#haiku.transform_with_state)
    that lets you use all of the hooks provided by Haiku plus some 
    custom hooks from Elegy:
    
    - [`add_loss`][elegy.hooks.add_loss]
    - [`add_metric`][elegy.hooks.add_metric]
    - [`add_summary`][elegy.hooks.add_summary]

    `transform.apply` return additional outputs that give you the structures
    collected by these hooks:

    ```python
    def f(a, b, c):
        ...
        elegy.add_loss("the_universe", 42)
        ...
    
    transform = elegy.transform(f)
    params, state = transform.init(rng, args=(a, b), kwargs={"c": c})  # f(a, b, c=c)
    outputs, state, losses, metrics, summaries = transform.apply(
        params, state, rng, args=(a, b), kwargs={"c": c}
    )

    assert losses["the_universe_loss"] == 42
    ``` 
    
    As in Haiku, the `rng` argument for `apply` is optional. Contrary
    to Haiku, `transform` is a class and not a function.

    Attributes:
        f: A function closing over [elegy.Module] instances.
    """

    f: tp.Callable

    def init(
        self,
        rng: tp.Optional[tp.Union[np.ndarray, int]],
        args: tp.Tuple = (),
        kwargs: tp.Optional[tp.Dict] = None,
    ) -> tp.Tuple[hk.Params, hk.State]:
        """
        Initializes your function collecting parameters and state.
        """
        if kwargs is None:
            kwargs = {}

        rng = src_transform.to_prng_sequence(rng, err_msg=src_transform.INIT_RNG_ERROR)

        with base.new_context(rng=rng) as ctx, elegy_context():
            self.f(*args, **kwargs)

        initital_state = ctx.collect_initial_state()

        return ctx.collect_params(), initital_state

    def apply(
        self,
        params: tp.Optional[hk.Params],
        state: tp.Optional[hk.State],
        rng: tp.Optional[tp.Union[np.ndarray, int]] = None,
        get_summaries: bool = False,
        args: tp.Tuple = (),
        kwargs: tp.Optional[tp.Dict] = None,
    ) -> TransformedState:
        """
        Applies your function injecting parameters and state.

        Arguments:
            params:
            state: 
            rng: 
            get_summaries: 
            args: 
            kwargs: 

        Returns:
            A [`TransformedState`][elegy.hooks.TransformedState] namedtuple consiting 
            of (outputs, state, losses, metrics, summaries).
        """
        if kwargs is None:
            kwargs = {}

        params = src_transform.check_mapping("params", params)
        state = src_transform.check_mapping("state", state)
        rng = src_transform.to_prng_sequence(
            rng,
            err_msg=(
                src_transform.APPLY_RNG_STATE_ERROR
                if state
                else src_transform.APPLY_RNG_ERROR
            ),
        )

        with base.new_context(
            params=params, state=state, rng=rng
        ) as ctx, elegy_context(get_summaries=get_summaries) as elegy_ctx:
            outputs = self.f(*args, **kwargs)

        return TransformedState(
            outputs=outputs,
            state=ctx.collect_state(),
            losses=elegy_ctx.losses,
            metrics=elegy_ctx.metrics,
            summaries=elegy_ctx.summaries,
        )


# def map_state(
#     f: tp.Callable[[tp.Tuple[str, ...], str, tp.Any], bool],
#     state: tp.Mapping,
#     parents: tp.Tuple = (),
# ):

#     output_state = {}

#     for key, value in state.items():

#         if isinstance(value, tp.Mapping):
#             value = map_state(f, value, parents + (key,))

#             if value:
#                 output_state[key] = value
#         else:
#             output_state[key] = f(parents, key, value)

#     return hk.data_structures.to_immutable_dict(output_state)
