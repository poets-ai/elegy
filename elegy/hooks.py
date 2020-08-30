import typing as tp

import jax.numpy as jnp
import numpy as np

from elegy import module
from elegy.module import LOCAL, Context, as_initial, get_unique_name
from elegy.types import PRNGKey

__all__ = [
    "add_loss",
    "add_metric",
    "add_summary",
    "get_parameter",
    "get_state",
    "next_rng_key",
    "set_state",
]


def get_parameter(
    name: str,
    shape: tp.Sequence[int] = (),
    dtype: tp.Optional[np.dtype] = None,
    initializer: tp.Union[
        tp.Callable[[tp.Sequence[int], tp.Any], tp.Any], tp.Any
    ] = jnp.zeros,
) -> np.ndarray:
    """
    A hook that lets you add a parameter to the current module. The parameter will only be created once
    during `init` and will reused afterwards.

    Arguments:
        name: The name of the parameter. It must be unique and no other field/property/method
            of the instance can have that name.
        shape: The shape of the parameter.
        dtype: The type of the parameter.
        initializer: A callable that takes in a shape and dtype and returns the initial value.

    Returns:
        The value of the parameter.
    """
    if LOCAL.contexts:
        context: Context = LOCAL.contexts[-1]
        module = context.module_c[-1]

        if not hasattr(module, name):
            if not context.building:
                raise ValueError(f"Trying to initialize '{name}' outside of `init`.")

            module._params.add(name)

            if dtype is None:
                dtype = module.dtype

            initial_value = (
                initializer(shape, dtype)
                if isinstance(initializer, tp.Callable)
                else initializer
            )

            setattr(module, name, initial_value)

        elif name not in module._params:
            raise ValueError(
                f"Class already contained a property named '{name}', "
                "please use a unique name for the parameter."
            )

        value = getattr(module, name)

        return value
    else:
        raise ValueError("Cannot execute `get_parameter` outside of a `elegy.context`")


def get_state(
    name: str,
    shape: tp.Sequence[int] = (),
    dtype: tp.Optional[np.dtype] = None,
    initializer: tp.Union[
        tp.Callable[[tp.Sequence[int], tp.Any], tp.Any], tp.Any
    ] = jnp.zeros,
) -> tp.Any:
    """
    A hook that lets you add a state to the current module. The state will only be created once
    during `init` and will reused afterwards.

    Arguments:
        name: The name of the state. It must be unique and no other field/property/method
            of the instance can have that name.
        shape: The shape of the state.
        dtype: The type of the state.
        initializer: A callable that takes in a shape and dtype and returns the initial value.

    Returns:
        The value of the state.
    """

    if LOCAL.contexts:
        context: Context = LOCAL.contexts[-1]
        module = context.module_c[-1]

        if not hasattr(module, name):
            if not context.building:
                raise ValueError(f"Trying to initialize '{name}' outside of `init`.")

            module._states.add(name)
            initial_name = as_initial(name)

            if dtype is None:
                dtype = module.dtype

            initial_value = (
                initializer(shape, dtype)
                if isinstance(initializer, tp.Callable)
                else initializer
            )

            setattr(module, name, initial_value)
            setattr(module, initial_name, initial_value)

        elif name not in module._states:
            raise ValueError(
                f"Class already contained a property named '{name}', "
                "please use a unique name for the state."
            )

        value = getattr(module, name)

        return value
    else:
        raise ValueError("Cannot execute `get_state` outside of a `elegy.context`")


def set_state(name: str, value: tp.Any) -> None:
    """
    A hook that lets you update a state of the current module, if the state does not
    exist it will be created.

    Arguments:
        name: The name of the state. It must be unique and no other field/property/method
            of the instance can have that name.
        value: The updated value of the state.
    """
    if LOCAL.contexts:
        context: Context = LOCAL.contexts[-1]
        module = context.module_c[-1]

        if name not in module._states:
            if not context.building:
                raise ValueError(f"Trying to initialize '{name}' outside of `init`.")

            module._states.add(name)
            initial_name = as_initial(name)

            setattr(module, name, value)
            setattr(module, initial_name, value)
        else:
            setattr(module, name, value)
    else:
        raise ValueError("Cannot execute `set_state` outside of a `elegy.context`")


def add_summary(name: tp.Optional[str], value: np.ndarray) -> None:
    """
    A hook that lets you define a summary in the current module. Its primary
    use is to keep track of certain values as they flow through the network
    so `Model.summary()` can show a representation of architecture.

    ```python
    def call(self, x):
        ...
        y = jax.nn.relu(x)
        elegy.add_summary("relu", y)
        ...
    ```

    The summaries will be aggregated by [`apply`][elegy.module.Module.apply]
    if `get_summaries` is set to `True`, else this hook does nothing.

    ```python
    transformed_state = transform.apply(..., get_summaries=True, ...)
    ```

    Arguments:
        name: The name of the loss. If a summary with the same
            `name` already exists a unique identifier will be generated.
        value: The value for the summary.
    """

    if LOCAL.contexts:
        context: Context = LOCAL.contexts[-1]
        module = context.module_c[-1]

        if not context.get_summaries:
            return

        # name = level_names[module]
        base_name = "/".join(context.path_names_c)

        base_name = f"{base_name}/{name}" if name is not None else base_name
        base_name = get_unique_name(context.summaries, base_name)
        module = module if name is None else None  # pass module only if name is None

        context.summaries.append((module, base_name, value))
    else:
        raise ValueError("Cannot execute `add_summary` outside of an `elegy.context`")


def add_loss(name: str, value: np.ndarray) -> None:
    """
    A hook that lets you define a loss within a [`module`][elegy.module.Module].

    ```python
    w = elegy.get_parameter("w", [3, 5], initializer=jnp.ones)

    # L2 regularization penalty
    elegy.add_loss("l2_regularization", 0.01 * jnp.mean(w ** 2))
    ```

    The loss will be aggregated by [`Module.apply`][elegy.module.Module.apply]
    and automatically handled by [`Model`][elegy.model.Model].

    Arguments:
        name: The name of the loss. If a `name` is repeated on
            different calls values will be added together.
        value: The value for the loss.
    """
    if not name.endswith("loss"):
        name += "_loss"

    if LOCAL.contexts:
        context: Context = LOCAL.contexts[-1]

        if name in context.losses:
            context.losses[name] += value
        else:
            context.losses[name] = value
    else:
        raise ValueError("Cannot execute `add_loss` outside of an `elegy.context`")


def add_metric(name: str, value: np.ndarray) -> None:
    """
    A hook that lets you define a metric within a [`module`][elegy.module.Module].

    ```python
    y = jax.nn.relu(x)
    elegy.add_metric("activation_mean", jnp.mean(y))
    ```

    The metrics will be aggregated by [`Module.apply`][elegy.module.Module.apply]
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
    A hook that returns a unique JAX RNG key split from the current global key.

    ```python
    key = elegy.next_rng_key()
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


def is_training() -> bool:
    """
    A hook that returns the current training status.

    ```python
    training = elegy.is_training()

    if training:
        ...
    else:
        ...
    ```

    Returns:
        A boolean value indicating whether training is currently happening.
    """
    if LOCAL.contexts:
        context: module.Context = module.LOCAL.contexts[-1]
        return context.training
    else:
        raise ValueError("Cannot execute `is_training` outside of an `elegy.context`")


# patch module
module.add_summary = add_summary
