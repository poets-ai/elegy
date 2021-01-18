from elegy.types import RNGSeq
from elegy import utils
from .generalized_optimizer import (
    GeneralizedOptimizer,
    Grads,
    NetParams,
    OptimizerStates,
    register_optimizer_for,
)
import typing as tp
import optax


@register_optimizer_for(optax.GradientTransformation)
class OptaxOptimizer(GeneralizedOptimizer):
    def __init__(self, optimizer: optax.GradientTransformation):
        self.optimizer = optimizer

    def init(self, rng: RNGSeq, net_params: NetParams) -> OptimizerStates:
        return self.optimizer.init(net_params)

    def apply(
        self,
        net_params: NetParams,
        grads: Grads,
        optimizer_states: OptimizerStates,
        rng: RNGSeq,
    ) -> tp.Tuple[NetParams, OptimizerStates]:
        updates, optimizer_states = self.optimizer.update(
            grads, optimizer_states, net_params
        )
        net_params = optax.apply_updates(net_params, updates)

        return net_params, optimizer_states
