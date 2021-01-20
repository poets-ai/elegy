from contextlib import contextmanager
from elegy.types import Logs, Path, RNGSeq, Scalar, Summaries
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
    rng: tp.Optional[RNGSeq]
    training: tp.Optional[bool]


@dataclass
class _HooksContext(threading.local):
    losses: tp.Optional[Logs]
    metrics: tp.Optional[Logs]
    summaries: tp.Optional[Summaries]
    rng: tp.Optional[RNGSeq]
    training: tp.Optional[bool]


LOCAL: HooksContext = _HooksContext(
    losses=None,
    metrics=None,
    summaries=None,
    rng=None,
    training=None,
)


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

    LOCAL.summaries.append((path, module, value))


def get_losses() -> tp.Optional[Logs]:
    return LOCAL.losses.copy() if LOCAL.losses is not None else None


def get_metrics() -> tp.Optional[Logs]:
    return LOCAL.metrics.copy() if LOCAL.metrics is not None else None


def get_summaries() -> tp.Optional[Summaries]:
    return LOCAL.summaries.copy() if LOCAL.summaries is not None else None


def summaries_active() -> bool:
    return LOCAL.summaries is not None


def get_rng() -> tp.Optional[RNGSeq]:
    return LOCAL.rng


def next_key() -> jnp.ndarray:
    if LOCAL.rng is None:
        raise ValueError(
            f"No rng present in context, please set it in `hooks_context`."
        )

    return LOCAL.rng.next()


def get_training() -> tp.Optional[bool]:
    return LOCAL.training


def is_training() -> bool:
    if LOCAL.training is None:
        raise ValueError(
            f"'training' not present in context, please set it in `hooks_context`."
        )

    return LOCAL.training


# ----------------------------------------------------------------
# contexts
# ----------------------------------------------------------------


def hooks_context(
    summaries: bool = False,
    rng: tp.Optional[RNGSeq] = None,
    training: tp.Optional[bool] = None,
) -> tp.ContextManager[None]:
    return _hooks_context(
        summaries=summaries,
        rng=rng,
        training=training,
    )


@contextmanager
def _hooks_context(
    summaries: bool,
    rng: tp.Optional[RNGSeq],
    training: tp.Optional[bool],
) -> tp.Iterator[None]:

    prev_losses = LOCAL.losses
    prev_metrics = LOCAL.metrics
    prev_summaries = LOCAL.summaries
    prev_rng = LOCAL.rng
    prev_training = LOCAL.training

    LOCAL.losses = {}
    LOCAL.metrics = {}
    LOCAL.summaries = [] if summaries else None
    LOCAL.rng = rng
    LOCAL.training = training

    try:
        yield
    finally:
        LOCAL.losses = prev_losses
        LOCAL.metrics = prev_metrics
        LOCAL.summaries = prev_summaries
        LOCAL.rng = prev_rng
        LOCAL.training = prev_training


# -------------------------------------------------------------
# transforms
# -------------------------------------------------------------


class TransformtOutput(tp.NamedTuple):
    output: tp.Any
    losses: tp.Optional[Logs]
    metrics: tp.Optional[Logs]
    summary_values: tp.Optional[tp.List[tp.Any]]
    rng: tp.Optional[RNGSeq]
    training: tp.Optional[bool]


class DynamicArgs(tp.NamedTuple):
    losses: tp.Optional[Logs]
    metrics: tp.Optional[Logs]
    summary_values: tp.Optional[tp.List[tp.Any]]
    rng: tp.Optional[RNGSeq]


class StaticArgs(tp.NamedTuple):
    training: tp.Optional[bool]


def _patch_summary_values(
    summaries: tp.Optional[Summaries],
    values: tp.Optional[tp.List[tp.Any]],
) -> tp.Optional[Summaries]:
    if values is not None and summaries is not None:
        return [
            (path, module, value) for (path, module, _), value in zip(summaries, values)
        ]
    else:
        assert values is None and summaries is None
        return None


def _extract_summary_values(
    summaries: tp.Optional[Summaries],
) -> tp.Optional[tp.List[tp.Any]]:
    if summaries is not None:
        return [value for path, module, value in summaries]
    else:
        return None


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

        static_args, dynamic_args = args[:2]  # get from beginning
        args = args[2:]

        (LOCAL.losses, LOCAL.metrics, summary_values, LOCAL.rng) = dynamic_args
        (LOCAL.training,) = static_args
        LOCAL.summaries = _patch_summary_values(LOCAL.summaries, summary_values)

        # call
        output = f(*args)

        # add outputs context
        return TransformtOutput(
            output=output,
            losses=get_losses(),
            metrics=get_metrics(),
            summary_values=_extract_summary_values(get_summaries()),
            rng=get_rng(),
            training=get_training(),
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
            losses=get_losses(),
            metrics=get_metrics(),
            summary_values=_extract_summary_values(get_summaries()),
            rng=get_rng(),
        )
        static_args = StaticArgs(
            training=get_training(),
        )
        # put them first because of static_args
        args = (static_args, dynamic_args) + args

        # call and patch
        (
            output,
            LOCAL.losses,
            LOCAL.metrics,
            summary_values,
            LOCAL.rng,
            training,
        ) = transform_fn(*args)
        LOCAL.summaries = _patch_summary_values(LOCAL.summaries, summary_values)

        if training is not None:
            assert training.dtype == jnp.dtype("bool")

        LOCAL.training = bool(training) if training is not None else None
        return output

    return wrapper


def value_and_grad(
    f,
    **kwargs,
) -> tp.Callable[..., tp.Tuple[tp.Any, tp.Any]]:
    def _transform_fn(
        *args,
    ) -> tp.Tuple[np.ndarray, TransformtOutput]:
        # extract input context
        dynamic_args: DynamicArgs
        static_args: StaticArgs

        dynamic_args, static_args = args[-2:]  # get from end
        args = args[:-2]

        (LOCAL.losses, LOCAL.metrics, summary_values, LOCAL.rng) = dynamic_args
        (LOCAL.training,) = static_args
        LOCAL.summaries = _patch_summary_values(LOCAL.summaries, summary_values)

        # call
        output = f(*args)
        loss = output[0] if isinstance(output, tuple) else output

        # add outputs context
        return loss, TransformtOutput(
            output=output,
            losses=get_losses(),
            metrics=get_metrics(),
            summary_values=_extract_summary_values(get_summaries()),
            rng=get_rng(),
            training=get_training(),
        )

    kwargs["has_aux"] = True
    transform_fn: tp.Callable[
        ..., tp.Tuple[tp.Tuple[np.ndarray, TransformtOutput], tp.Any]
    ] = jax.value_and_grad(_transform_fn, **kwargs)

    @functools.wraps(f)
    def wrapper(*args):
        # add input context
        dynamic_args = DynamicArgs(
            losses=get_losses(),
            metrics=get_metrics(),
            summary_values=_extract_summary_values(get_summaries()),
            rng=get_rng(),
        )
        static_args = StaticArgs(
            training=get_training(),
        )
        # put them last because params have to go first
        args += (dynamic_args, static_args)

        # call and patch
        (
            (
                loss,
                (
                    output,
                    LOCAL.losses,
                    LOCAL.metrics,
                    summary_values,
                    LOCAL.rng,
                    LOCAL.training,
                ),
            ),
            grads,
        ) = transform_fn(*args)
        LOCAL.summaries = _patch_summary_values(LOCAL.summaries, summary_values)

        return output, grads

    return wrapper


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
