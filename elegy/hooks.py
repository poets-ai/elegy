import threading
import typing as tp
from contextlib import contextmanager

import haiku as hk
import numpy as np
from haiku._src import base

LOCAL = threading.local()
LOCAL.calculating_summary = False
LOCAL.layer_outputs = None

T = tp.TypeVar("T")


class LocalList(list, threading.local):
    pass


CONTEXTS = LocalList()
RNG_ERROR_TPL = (
    "{f} must be called with an RNG as the {position} argument, "
    "the required signature is: `{signature}`"
)
INIT_RNG_ERROR = RNG_ERROR_TPL.format(
    f="Init", position="first", signature="init(rng, *a, **k)"
)
APPLY_RNG_ERROR = RNG_ERROR_TPL.format(
    f="Apply", position="second", signature="apply(params, rng, *a, **k)"
)
APPLY_RNG_STATE_ERROR = RNG_ERROR_TPL.format(
    f="Apply", position="third", signature="apply(params, state, rng, *a, **k)"
)


@contextmanager
def elegy_hooks():
    context = {}
    CONTEXTS.append(context)
    try:
        yield context
    finally:
        CONTEXTS.pop()


def log_value(topic, name, value):
    # NOTE: log(..) ignored when not logging.
    if CONTEXTS:
        context = CONTEXTS[-1]

        if topic in context:
            context[topic][name] = value
        else:
            context[topic] = {name: value}


class layer_summaries:
    def __enter__(self):
        self.calculating_summary = LOCAL.calculating_summary
        self.layer_outputs = LOCAL.layer_outputs

        LOCAL.calculating_summary = True
        LOCAL.layer_outputs = []

    def __exit__(self, *args):
        LOCAL.calculating_summary = self.calculating_summary
        LOCAL.layer_outputs = self.layer_outputs


class no_op:
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


class TransformedState(tp.NamedTuple):
    outputs: tp.Any
    state: hk.State
    losses: hk.State
    metrics: hk.State
    layer_outputs: hk.State


class Transform(tp.NamedTuple):
    f: tp.Callable

    def init(
        self, rng: tp.Optional[tp.Union[np.ndarray, int]], *args, **kwargs,
    ) -> tp.Tuple[hk.Params, hk.State]:
        """
        Initializes your function collecting parameters and state.
        """

        rng = to_prng_sequence(rng, err_msg=INIT_RNG_ERROR)

        with base.new_context(rng=rng) as ctx:
            self.f(*args, **kwargs)

        initital_state = ctx.collect_initial_state()

        return ctx.collect_params(), initital_state

    def apply(
        self,
        params: tp.Optional[hk.Params],
        state: tp.Optional[hk.State],
        rng: tp.Optional[tp.Union[np.ndarray, int]],
        *args,
        **kwargs,
    ) -> TransformedState:
        """
        Applies your function injecting parameters and state.
        """

        params = check_mapping("params", params)
        state = check_mapping("state", state)
        rng = to_prng_sequence(
            rng, err_msg=(APPLY_RNG_STATE_ERROR if state else APPLY_RNG_ERROR)
        )

        with base.new_context(params=params, state=state, rng=rng) as ctx:
            outputs = self.f(*args, **kwargs)

        state = ctx.collect_state()
        losses = {}
        metrics = {}

        return TransformedState(
            outputs=outputs,
            state=state,
            losses=losses,
            metrics=metrics,
            layer_outputs=dict(LOCAL.layer_outputs) if LOCAL.layer_outputs else {},
        )


def transform(f) -> Transform:
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

    return Transform(f)


def to_prng_sequence(rng, err_msg) -> tp.Optional[hk.PRNGSequence]:
    if rng is not None:
        try:
            rng = hk.PRNGSequence(rng)
        except Exception as e:
            raise ValueError(err_msg) from e
    return rng


def check_mapping(name: str, mapping: tp.Optional[T]) -> T:
    """Cleans inputs to apply_fn, providing better errors."""
    # TODO(tomhennigan) Remove support for empty non-Mappings.
    if mapping is None:
        # Convert None to empty dict.
        mapping = dict()
    if not isinstance(mapping, tp.Mapping):
        raise TypeError(
            f"{name} argument does not appear valid: {mapping!r}. "
            "For reference the parameters for apply are "
            "`apply(params, rng, ...)`` for `hk.transform` and "
            "`apply(params, state, rng, ...)` for "
            "`hk.transform_with_state`."
        )
    return mapping


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
