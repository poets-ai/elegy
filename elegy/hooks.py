from contextlib import contextmanager
from elegy.types import Logs, Path, Scalar, Summaries
from elegy import utils
import functools
import threading
import typing as tp
from dataclasses import dataclass
import jax

import jax.numpy as jnp
import numpy as np

from elegy.types import Protocol


class HooksContext(Protocol):
    losses: tp.Optional[Logs]
    metrics: tp.Optional[Logs]
    summaries: tp.Optional[Summaries]


@dataclass
class _HooksContext(threading.local):
    losses: tp.Optional[Logs]
    metrics: tp.Optional[Logs]
    summaries: tp.Optional[Summaries]


LOCAL: HooksContext = _HooksContext(losses=None, metrics=None, summaries=None)


# ----------------------------------------------------------------
# hooks
# ----------------------------------------------------------------


def add_loss(name: str, value: Scalar) -> None:
    """
    A hook that lets you define a loss within a [`module`][elegy.module.Module].

    ```python
    w = self.add_parameter("w", [3, 5], initializer=jnp.ones)

    # L2 regularization penalty
    elegy.add_loss("l2_regularization", 0.01 * jnp.mean(w ** 2))
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


def add_metric(name: str, value: Scalar) -> None:
    """
    A hook that lets you define a metric within a [`module`][elegy.module.Module].

    ```python
    y = jax.nn.relu(x)
    elegy.add_metric("activation_mean", jnp.mean(y))
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
    path: Path,
    name: str,
    module: tp.Any,
    value: Scalar,
) -> None:
    """
    A hook that lets you define a summary in the current module. Its primary
    use is to keep track of certain values as they flow through the network
    so [`Model.summary`][elegy.model.model.Model.summary] can show a representation of architecture.

    ```python
    def call(self, x):
        ...
        y = jax.nn.relu(x)
        elegy.add_summary("relu", y)
        ...
    ```

    Arguments:
        module_or_name: The name of the summary or alternatively the module that this summary will represent.
            If a summary with the same name already exists a unique identifier will be generated.
        value: The value for the summary.
    """

    if LOCAL.summaries is None:
        return

    names = {"/".join(map(str, path)) for path in LOCAL.summaries.keys()}
    name = utils.get_unique_name(names, name)

    path += (name,)

    LOCAL.summaries[path] = (module, value)


def get_losses() -> tp.Optional[Logs]:
    return LOCAL.losses.copy() if LOCAL.losses is not None else None


def get_metrics() -> tp.Optional[Logs]:
    return LOCAL.metrics.copy() if LOCAL.metrics is not None else None


def get_summaries() -> tp.Optional[Summaries]:
    return LOCAL.summaries.copy() if LOCAL.summaries is not None else None


# -------------------------------------------------------------
# transforms
# -------------------------------------------------------------


class TransformtOutput(tp.NamedTuple):
    output: tp.Any
    losses: tp.Optional[Logs]
    metrics: tp.Optional[Logs]
    summary_values: tp.Optional[tp.List[tp.Any]]


def hooks_aware(jax_f):
    @functools.wraps(jax_f)
    def _jax_transform(
        f,
        **kwargs,
    ):
        def _transform_fn(
            *args,
        ) -> TransformtOutput:

            output = f(*args)
            losses = get_losses()
            metrics = get_metrics()
            summaries = get_summaries()

            summary_values = (
                [value for module, value in summaries.values()]
                if summaries is not None
                else None
            )

            return TransformtOutput(
                output=output,
                losses=losses,
                metrics=metrics,
                summary_values=summary_values,
            )

        transform_fn = jax.jit(_transform_fn, **kwargs)

        @functools.wraps(f)
        def wrapper(*args):

            output, losses, metrics, summary_values = transform_fn(*args)

            summaries = {}

            if summary_values is not None:
                for key, value in zip(LOCAL.summaries.keys(), summary_values):
                    module = LOCAL.summaries[key][0]
                    summaries[key] = (module, value)

            LOCAL.losses = losses
            LOCAL.metrics = metrics
            LOCAL.summaries = summaries

            return output

        return wrapper

    return _jax_transform


jit = hooks_aware(jax.jit)

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
# vmap = hooks_aware(jax.vmap)
# pmap = hooks_aware(jax.pmap)
# soft_pmap = hooks_aware(jax.soft_pmap)


def value_and_grad(
    f,
    **kwargs,
) -> tp.Callable[..., tp.Tuple[tp.Any, tp.Any]]:
    def _transform_fn(
        *args,
    ) -> tp.Tuple[np.ndarray, TransformtOutput]:

        output = f(*args)

        losses = get_losses()
        metrics = get_metrics()
        summaries = get_summaries()
        summary_values = (
            [value for module, value in summaries.values()]
            if summaries is not None
            else None
        )

        loss = output[0] if isinstance(output, tuple) else output

        return loss, TransformtOutput(
            output=output,
            losses=losses,
            metrics=metrics,
            summary_values=summary_values,
        )

    kwargs["has_aux"] = True
    transform_fn: tp.Callable[
        ..., tp.Tuple[tp.Tuple[np.ndarray, TransformtOutput], tp.Any]
    ] = jax.value_and_grad(_transform_fn, **kwargs)

    @functools.wraps(f)
    def wrapper(*args):

        (loss, (output, losses, metrics, summary_values)), grads = transform_fn(*args)

        summaries = {}

        if summary_values is not None:
            for key, value in zip(LOCAL.summaries.keys(), summary_values):
                module = LOCAL.summaries[key][0]
                summaries[key] = (module, value)

        LOCAL.losses = losses
        LOCAL.metrics = metrics
        LOCAL.summaries = summaries

        return output, grads

    return wrapper


# ----------------------------------------------------------------
# contexts
# ----------------------------------------------------------------


def hooks_context() -> tp.ContextManager[None]:
    return _hooks_context()


@contextmanager
def _hooks_context() -> tp.Iterator[None]:

    prev_losses = LOCAL.losses
    prev_metrics = LOCAL.metrics
    prev_summaries = LOCAL.summaries

    LOCAL.losses = {}
    LOCAL.metrics = {}
    LOCAL.summaries = {}

    try:
        yield
    finally:
        LOCAL.losses = prev_losses
        LOCAL.metrics = prev_metrics
        LOCAL.summaries = prev_summaries
