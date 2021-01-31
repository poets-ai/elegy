import typing as tp

import jax.numpy as jnp
import numpy as np
import optax

from elegy import module, types, utils
from elegy.generalized_optimizer.generalized_optimizer import GeneralizedOptimizer


class LRScheduler(types.Protocol):
    def __call__(
        self, step: jnp.ndarray, epoch: tp.Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        ...


class Optimizer(GeneralizedOptimizer):
    r"""A Module that wraps around `optax` optimizers."""

    def __init__(
        self,
        *optimizer: optax.GradientTransformation,
        lr_schedule: tp.Optional[LRScheduler] = None,
        steps_per_epoch: tp.Union[int, jnp.ndarray, np.ndarray, None] = None,
        **kwargs,
    ):
        r"""
        Arguments:
            optimizer: An optax `GradientTransformation` object, if more than one is passed via `*args` then they are
                grouped using `optax.chain`.
            lr_schedule: A optional callable of the form `def lr_schedule(step: int, epoch: Optional[int]) -> float` that
                returns the learning rate schedule at each time step. If `steps_per_epoch` is given then epoch is calculated,
                else epoch is None.
            steps_per_epoch: The number of steps to in an epoch, needed to caculate `epoch` from `step`.
        """

        if len(optimizer) == 0:
            raise ValueError("Must pass atleast 1 optimizer, got 0")

        elif lr_schedule is not None:
            # do this to preserve reference after re-assign latter
            base_schedule = lr_schedule

            def lr_schedule_(step: jnp.ndarray) -> jnp.ndarray:
                epoch: tp.Any = (
                    step // steps_per_epoch if steps_per_epoch is not None else None
                )

                return base_schedule(step, epoch)

            optimizer = optax.chain(
                *optimizer,
                optax.scale_by_schedule(lr_schedule_),
            )

            lr_schedule = lr_schedule_

        elif len(optimizer) == 1:
            optimizer = optimizer[0]
        else:
            optimizer = optax.chain(*optimizer)

        self.optimizer = optimizer
        self.lr_schedule = lr_schedule

    def init(
        self, rng: types.RNGSeq, net_params: types.NetParams
    ) -> types.OptimizerStates:
        return self.optimizer.init(net_params)

    def apply(
        self,
        net_params: types.NetParams,
        grads: types.Grads,
        optimizer_states: types.OptimizerStates,
        rng: types.RNGSeq,
    ) -> tp.Tuple[types.NetParams, types.OptimizerStates]:
        updates, optimizer_states = self.optimizer.update(
            grads, optimizer_states, net_params
        )
        net_params = optax.apply_updates(net_params, updates)

        return net_params, optimizer_states

    def current_lr(
        self, optimizer_states: types.OptimizerStates
    ) -> tp.Optional[jnp.ndarray]:
        """Returns the learning rate scaled by schedule(s) that will be used for the next training step"""

        if (
            not isinstance(optimizer_states, types.Uninitialized)
            and self.lr_schedule is not None
        ):
            step = optimizer_states[-1].count
            return self.lr_schedule(step)
