import typing as tp

import jax
import jax.numpy as jnp
import optax

from elegy.pytree import PytreeObject, static_field

O = tp.TypeVar("O", bound="Optimizer")
A = tp.TypeVar("A", bound="tp.Any")


class Optimizer(PytreeObject):
    """Wraps an optax optimizer and turn it into a Pytree while maintaining a similar API.

    The main difference with optax is that Optimizer contains its own state, thus, there is
    no `opt_state`.

    Example:
    ```python
    def main():
        ...
        optimizer = Optimizer(optax.adam(1e-3))
        optimizer = optimizer.init(params)
        ...

    jax.jit
    def train_step(model, x, y, optimizer):
        ...
        params = optimizer.update(grads, params)
        ...
        return model, loss, optimizer
    ```

    Notice that since the optimizer is a `Pytree` it can naturally pass through `jit`.

    ### Differences with Optax
    * `init` return a new optimizer instance, there is no `opt_state`.
    * `update` doesn't get `opt_state` as an argument, instead it performs updates
        to its internal state.
    * `update` applies the updates to the params and returns them by default, use `update=False` to
        to get the param updates instead.

    Arguments:
        optimizer: An optax optimizer.
    """

    optimizer: optax.GradientTransformation
    opt_state: tp.Optional[tp.Any] = None
    _n_params: tp.Optional[int] = static_field(None)

    def __init__(self, optimizer: optax.GradientTransformation) -> None:
        self.optimizer = optimizer

    def init(self: O, params: tp.Any) -> O:
        """
        Initialize the optimizer from an initial set of parameters.

        Arguments:
            params: An initial set of parameters.

        Returns:
            A new optimizer instance.
        """
        params = jax.tree_leaves(params)

        return self.replace(
            opt_state=self.optimizer.init(params),
            _n_params=len(params),
        )

    # NOTE: params are flattened because:
    # - The flat list is not a Module, thus all of its internal parameters in the list are marked as
    # OptState by a single annotation (no need to rewrite the module's annotations)
    # - It ignores the static part of Modules which if changed Optax yields an error.
    def update(
        self: O, grads: A, params: tp.Optional[A] = None, apply_updates: bool = True
    ) -> tp.Tuple[A, O]:
        """
        Applies the parameters updates and updates the optimizers internal state.

        Arguments:
            grads: the gradients to perform the update.
            params: the parameters to update. If `None` then `update` has to be `False`.
            apply_updates: if `False` then the updates are returned instead of being applied.

        Returns:
            A (params, optimizer) tuple. If `apply_updates` is `False` then the updates
            to the params are returned instead of being applied.
        """
        if self.opt_state is None:
            raise RuntimeError("Optimizer is not initialized")

        assert self.opt_state is not None
        if apply_updates and params is None:
            raise ValueError("params must be provided if updates are being applied")

        opt_grads, treedef = jax.tree_flatten(grads)
        opt_params = jax.tree_leaves(params)

        if len(opt_params) != self._n_params:
            raise ValueError(
                f"params must have length {self._n_params}, got {len(opt_params)}"
            )
        if len(opt_grads) != self._n_params:
            raise ValueError(
                f"grads must have length {self._n_params}, got {len(opt_grads)}"
            )

        param_updates: A
        param_updates, opt_state = self.optimizer.update(
            opt_grads, self.opt_state, opt_params
        )

        output: A
        if apply_updates:
            output = optax.apply_updates(opt_params, param_updates)
        else:
            output = param_updates

        output = jax.tree_unflatten(treedef, output)

        return output, self.replace(opt_state=opt_state)
