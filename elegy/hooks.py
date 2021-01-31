from contextlib import contextmanager
from elegy import utils, types
import functools
import threading
import typing as tp
from dataclasses import dataclass
import jax

import jax.numpy as jnp
import numpy as np

from elegy import types

__all__ = [
    "add_loss",
    "add_metric",
    "add_summary",
    "get_losses",
    "losses_active",
    "get_metrics",
    "metrics_active",
    "get_summaries",
    "summaries_active",
    "context",
    "jit",
]


class HooksContext(types.Protocol):
    losses: tp.Optional[types.Logs]
    metrics: tp.Optional[types.Logs]
    summaries: tp.Optional[types.Summaries]


@dataclass
class _HooksContext(threading.local):
    losses: tp.Optional[types.Logs]
    metrics: tp.Optional[types.Logs]
    summaries: tp.Optional[types.Summaries]


LOCAL: HooksContext = _HooksContext(
    losses=None,
    metrics=None,
    summaries=None,
)


# ----------------------------------------------------------------
# hooks
# ----------------------------------------------------------------


def add_loss(name: str, value: types.Scalar) -> None:
    """
    A hook that lets you define a loss within a [`module`][elegy.module.Module].

    ```python
    w = self.add_parameter("w", [3, 5], initializer=jnp.ones)

    # L2 regularization penalty
    elegy.hooks.add_loss("l2_regularization", 0.01 * jnp.mean(w ** 2))
    ```

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
        LOCAL.losses[name] = LOCAL.losses[name] + value
    else:
        LOCAL.losses[name] = value


def add_metric(name: str, value: types.Scalar) -> None:
    """
    A hook that lets you define a metric within a [`module`][elegy.module.Module].

    ```python
    y = jax.nn.relu(x)
    elegy.hooks.add_metric("activation_mean", jnp.mean(y))
    ```

    Arguments:
        name: The name of the loss. If a metric with the same
            `name` already exists a unique identifier will be generated.
        value: The value for the metric.
    """
    if LOCAL.metrics is None:
        return

    # name = f"{base_name()}/{name}"
    name = utils.get_unique_name(set(LOCAL.metrics), name)
    LOCAL.metrics[name] = value


def add_summary(
    path: types.Path,
    module: tp.Any,
    value: tp.Any,
) -> None:
    """
    A hook that lets you define a summary in the current module. Its primary
    use is to keep track of certain values as they flow through the network
    so [`Model.summary`][elegy.model.model.Model.summary] can show a representation of architecture.

    ```python
    def call(self, x):
        ...
        y = jax.nn.relu(x)
        elegy.hooks.add_summary("relu", y)
        ...
    ```

    Arguments:
        module_or_name: The name of the summary or alternatively the module that this summary will represent.
            If a summary with the same name already exists a unique identifier will be generated.
        value: The value for the summary.
    """

    if not summaries_active():
        return

    LOCAL.summaries.append(types.Summary(path, module, value))


def get_losses() -> types.Logs:
    if LOCAL.losses is None:
        return {}

    return LOCAL.losses.copy()


def losses_active() -> bool:
    return LOCAL.losses is not None


def get_metrics() -> types.Logs:
    if LOCAL.metrics is None:
        return {}

    return LOCAL.metrics.copy()


def metrics_active() -> bool:
    return LOCAL.metrics is not None


def get_summaries() -> types.Summaries:
    if LOCAL.summaries is None:
        return []

    return LOCAL.summaries.copy()


def summaries_active() -> bool:
    return LOCAL.summaries is not None


# ----------------------------------------------------------------
# contexts
# ----------------------------------------------------------------


def context(
    *,
    losses: tp.Union[types.Logs, bool, None] = None,
    metrics: tp.Union[types.Logs, bool, None] = None,
    summaries: tp.Union[types.Summaries, bool, None] = None,
    set_all: bool = False,
) -> tp.ContextManager[None]:

    if set_all:
        if losses is None:
            losses = True
        if metrics is None:
            metrics = True
        if summaries is None:
            summaries = True

    if isinstance(losses, bool):
        losses = {} if losses else None

    if isinstance(metrics, bool):
        metrics = {} if metrics else None

    if isinstance(summaries, bool):
        summaries = [] if summaries else None

    return _context(
        losses=losses,
        metrics=metrics,
        summaries=summaries,
    )


@contextmanager
def _context(
    losses: tp.Optional[types.Logs],
    metrics: tp.Optional[types.Logs],
    summaries: tp.Optional[types.Summaries],
) -> tp.Iterator[None]:

    prev_losses = LOCAL.losses
    prev_metrics = LOCAL.metrics
    prev_summaries = LOCAL.summaries

    LOCAL.losses = losses
    LOCAL.metrics = metrics
    LOCAL.summaries = summaries

    try:
        yield
    finally:
        LOCAL.losses = prev_losses
        LOCAL.metrics = prev_metrics
        LOCAL.summaries = prev_summaries


# -------------------------------------------------------------
# transforms
# -------------------------------------------------------------


class TransformtOutput(tp.NamedTuple):
    output: tp.Any
    losses: tp.Optional[types.Logs]
    metrics: tp.Optional[types.Logs]
    summaries: tp.Optional[tp.List[tp.Any]]


class DynamicArgs(tp.NamedTuple):
    losses: tp.Optional[types.Logs]
    metrics: tp.Optional[types.Logs]
    summaries: tp.Optional[tp.List[tp.Any]]


class StaticArgs(tp.NamedTuple):
    pass


def _update_local_context(
    losses: tp.Optional[types.Logs],
    metrics: tp.Optional[types.Logs],
    summaries: tp.Optional[types.Summaries],
):
    if LOCAL.losses is not None and losses is not None:
        LOCAL.losses.clear()
        LOCAL.losses.update(losses)

    if LOCAL.metrics is not None and metrics is not None:
        LOCAL.metrics.clear()
        LOCAL.metrics.update(metrics)

    if LOCAL.summaries is not None and summaries is not None:
        LOCAL.summaries.clear()
        LOCAL.summaries.extend(summaries)


def jit(
    f,
    **kwargs,
):
    def _transform_fn(
        *args,
    ) -> TransformtOutput:

        # extract input context
        dynamic_args: DynamicArgs
        static_args: StaticArgs

        # static_args is unused because they dont need to be set back
        static_args, dynamic_args = args[:2]  # get from beginning
        args = args[2:]

        (losses, metrics, summaries) = dynamic_args

        # perform updates
        _update_local_context(losses, metrics, summaries)

        # call
        output = f(*args)

        # add outputs context
        return TransformtOutput(
            output=output,
            losses=LOCAL.losses,
            metrics=LOCAL.metrics,
            summaries=LOCAL.summaries,
        )

    # transform kwargs
    static_argnums = kwargs.pop("static_argnums", [])
    if isinstance(static_argnums, int):
        static_argnums = [static_argnums]
    static_argnums = [0] + [i + 2 for i in static_argnums]

    # make fn
    transform_fn = jax.jit(
        _transform_fn,
        static_argnums=static_argnums,
        **kwargs,
    )

    @functools.wraps(f)
    def wrapper(*args):

        # add input context
        dynamic_args = DynamicArgs(
            losses=LOCAL.losses,
            metrics=LOCAL.metrics,
            summaries=LOCAL.summaries,
        )
        static_args = StaticArgs()
        # put them first because of static_args
        args = (static_args, dynamic_args) + args

        # call and patch
        (
            output,
            losses,
            metrics,
            summaries,
        ) = transform_fn(*args)

        # perform updates
        _update_local_context(losses, metrics, summaries)

        return output

    return wrapper


def hooks_aware(transform):
    def transform_wrapper(
        f,
        **kwargs,
    ):
        def f_wrapper(
            *args,
        ) -> TransformtOutput:

            # extract input context
            dynamic_args: DynamicArgs

            # static_args is unused because they dont need to be set back
            dynamic_args = args[0]  # get from beginning
            args = args[1:]

            (losses, metrics, summaries) = dynamic_args

            # perform updates
            _update_local_context(losses, metrics, summaries)

            # call
            output = f(*args)

            # add outputs context
            return TransformtOutput(
                output=output,
                losses=LOCAL.losses,
                metrics=LOCAL.metrics,
                summaries=LOCAL.summaries,
            )

        # make fn
        transform_fn = transform(
            f_wrapper,
            **kwargs,
        )

        @functools.wraps(f)
        def wrapper(*args):

            # add input context
            dynamic_args = DynamicArgs(
                losses=LOCAL.losses,
                metrics=LOCAL.metrics,
                summaries=LOCAL.summaries,
            )
            # put them first because of static_args
            args = (dynamic_args,) + args

            # call and patch
            (
                output,
                losses,
                metrics,
                summaries,
            ) = transform_fn(*args)

            # perform updates
            _update_local_context(losses, metrics, summaries)

            return output

        return wrapper

    return transform_wrapper


# NOTE: it is unclear if these can be implemented since they dont support `hax_aux`
# jacrev = hooks_aware(jax.jacrev)
# hessian = hooks_aware(jax.hessian)
# mask = hooks_aware(jax.mask)
# jvp = hooks_aware(jax.jvp)
# linearize = hooks_aware(jax.linearize)
# vjp = hooks_aware(jax.vjp)
# linear_transpose = hooks_aware(jax.linear_transpose)
# make_jaxpr = hooks_aware(jax.make_jaxpr)
# defjvp_all = hooks_aware(jax.defjvp_all)
# defjvp = hooks_aware(jax.defjvp)
# defvjp_all = hooks_aware(jax.defvjp_all)
# defvjp = hooks_aware(jax.defvjp)


# NOTE: these can work but require special handling of the axis dimension
vmap = hooks_aware(jax.vmap)
pmap = hooks_aware(jax.pmap)
# soft_pmap = hooks_aware(jax.soft_pmap)
