from contextlib import contextmanager
from elegy.types import Logs, Scalar
import functools
import threading
import typing as tp
from dataclasses import dataclass
import jax

import jax.numpy as jnp
import numpy as np

from elegy.utils import Protocol


class HooksContext(Protocol):
    losses: tp.Optional[Logs]
    metrics: tp.Optional[Logs]


@dataclass
class _HooksContext(threading.local):
    losses: tp.Optional[Logs]
    metrics: tp.Optional[Logs]


LOCAL: HooksContext = _HooksContext(losses=None, metrics=None)


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
        LOCAL.losses[name] += value
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
    name = get_unique_name(set(LOCAL.metrics), name)
    LOCAL.metrics[name] = value


def get_losses() -> tp.Optional[Logs]:
    return LOCAL.losses


def get_metrics() -> tp.Optional[Logs]:
    return LOCAL.metrics


def get_total_loss() -> np.ndarray:
    losses = get_losses()
    loss = sum(losses.values(), np.ndarray(0.0))
    return loss


# -------------------------------------------------------------
# transforms
# -------------------------------------------------------------


class TransformtOutput(tp.NamedTuple):
    output: tp.Any
    losses: tp.Dict[str, np.ndarray]
    metrics: tp.Dict[str, np.ndarray]


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

            assert isinstance(losses, tp.Dict)
            assert isinstance(metrics, tp.Dict)

            return TransformtOutput(
                output=output,
                losses=losses,
                metrics=metrics,
            )

        transform_fn = jax.jit(_transform_fn, **kwargs)

        @functools.wraps(f)
        def wrapper(*args):

            output, losses, metrics = transform_fn(*args)

            LOCAL.losses = losses
            LOCAL.metrics = metrics

            return output

        return wrapper

    return _jax_transform


jit = hooks_aware(jax.jit)


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

        assert isinstance(losses, tp.Dict)
        assert isinstance(metrics, tp.Dict)

        loss = output[0] if isinstance(output, tuple) else output

        return loss, TransformtOutput(
            output=output,
            losses=losses,
            metrics=metrics,
        )

    kwargs["has_aux"] = True
    transform_fn: tp.Callable[
        ..., tp.Tuple[np.ndarray, TransformtOutput]
    ] = jax.value_and_grad(_transform_fn, **kwargs)

    @functools.wraps(f)
    def wrapper(*args):

        _, (output, losses, metrics) = transform_fn(*args)

        LOCAL.losses = losses
        LOCAL.metrics = metrics

        return output

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

    LOCAL.losses = {}
    LOCAL.metrics = {}

    try:
        yield
    finally:
        LOCAL.losses = prev_losses
        LOCAL.metrics = prev_metrics


# ----------------------------------------------------------------
# utils
# ----------------------------------------------------------------


def get_unique_name(
    names: tp.Set[str],
    name: str,
):

    if name not in names:
        return name

    i = 1
    while f"{name}_{i}" in names:
        i += 1

    return f"{name}_{i}"
