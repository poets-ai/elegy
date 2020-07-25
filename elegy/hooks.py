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
    summaries: tp.Dict[str, tp.Any]

    @classmethod
    def create(cls, get_summaries: bool = False):
        return cls(get_summaries=get_summaries, losses={}, metrics={}, summaries={})


@contextmanager
def elegy_context(get_summaries: bool = False):

    if get_summaries is None:
        context_get_summaries: bool = LOCAL.get_summariess
        get_summaries = context_get_summaries

    context = Context.create(get_summaries=get_summaries)
    LOCAL.contexts.append(context)
    try:
        yield context
    finally:
        LOCAL.contexts.pop()


def add_loss(name: str, value: np.ndarray):
    name += "_loss"
    if LOCAL.contexts:
        context: Context = LOCAL.contexts[-1]

        if name in context.losses:
            context.losses[name] += value
        else:
            context.losses[name] = value
    else:
        raise ValueError("Cannot execute `add_loss` outside of a `elegy.transform`")


def add_metric(name: str, value: np.ndarray):
    if LOCAL.contexts:
        context: Context = LOCAL.contexts[-1]

        name = f"{base.current_bundle_name()}/{name}"
        name = get_unique_name(context.summaries, name)

        context.metrics[name] = value
    else:
        raise ValueError("Cannot execute `add_loss` outside of a `elegy.transform`")


def add_summary(name: str, value: np.ndarray):
    if LOCAL.contexts:
        context: Context = LOCAL.contexts[-1]

        if not context.get_summaries:
            return

        name = f"{base.current_bundle_name()}/{name}"
        name = get_unique_name(context.summaries, name)

        context.summaries[name] = value
    else:
        raise ValueError("Cannot execute `add_loss` outside of a `elegy.transform`")


def get_unique_name(logs, name):

    if name not in logs:
        return name

    i = 1
    while f"{name}_{i}" in logs:
        i += 1

    return f"{name}_{i}"


class TransformedState(tp.NamedTuple):
    outputs: tp.Any
    state: hk.State
    losses: hk.State
    metrics: hk.State
    summaries: hk.State


class transform(tp.NamedTuple):
    """
    Transforms a function using Haiku modules into a pair of pure functions.

    See :func:`transform` for general details on Haiku transformations.
    For a function ``out = f(*a, **k)`` this function returns a pair of two pure
    functions that call ``f(*a, **k)`` explicitly collecting and injecting
    parameter values and state::

    ```python
    transform = elegy.transform(f)
    params, state = transform.init(rng, *a, **k)
    out, state = transform.apply(params, state, rng, *a, **k)
    ```
    
    Note that the ``rng`` argument is typically not required for ``apply`` and
    passing ``None`` is accepted.
    This function is equivalent to :func:`transform`, however it allows you to
    maintain and update internal state (e.g. :class:`ExponentialMovingAverage` in
    :class:`BatchNorm`) via :func:`get_state` and :func:`set_state`:
    >>> def f():
    ...   counter = hk.get_state("counter", shape=[], dtype=jnp.int32,
    ...                          init=jnp.zeros)
    ...   hk.set_state("counter", counter + 1)
    ...   return counter
    >>> f = hk.transform_with_state(f)
    >>> params, state = f.init(None)
    >>> for _ in range(10):
    ...   counter, state = f.apply(params, state, None)
    >>> counter
    DeviceArray(9, dtype=int32)

    Arguments:
        f: A function closing over :class:`Module` instances.
    
    Returns:
        A :class:`TransformedWithState` tuple with ``init`` and ``apply`` pure
        functions.
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
        rng: tp.Optional[tp.Union[np.ndarray, int]],
        get_summaries: bool = False,
        args: tp.Tuple = (),
        kwargs: tp.Optional[tp.Dict] = None,
    ) -> TransformedState:
        """
        Applies your function injecting parameters and state.
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
