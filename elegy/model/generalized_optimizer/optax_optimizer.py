import typing as tp

import optax
from elegy import types, utils

from .generalized_optimizer import GeneralizedOptimizer, register_optimizer_for


@register_optimizer_for(optax.GradientTransformation)
class OptaxOptimizer(GeneralizedOptimizer):
    def __init__(self, optimizer: optax.GradientTransformation):
        self.optimizer = optimizer

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
