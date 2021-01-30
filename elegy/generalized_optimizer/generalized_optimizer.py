import typing as tp
from abc import ABC, abstractmethod

from elegy import types, utils

REGISTRY: tp.Dict[tp.Type, tp.Type["GeneralizedOptimizer"]] = {}


class ModuleExists(Exception):
    pass


class GeneralizedOptimizer(ABC):
    @abstractmethod
    def __init__(self, optimizer: tp.Any):
        ...

    @abstractmethod
    def init(
        self, rng: types.RNGSeq, net_params: types.NetParams
    ) -> types.OptimizerStates:
        ...

    @abstractmethod
    def apply(
        self,
        net_params: types.NetParams,
        grads: types.Grads,
        optimizer_states: types.OptimizerStates,
        rng: types.RNGSeq,
    ) -> tp.Tuple[types.NetParams, types.OptimizerStates]:
        ...


def register_optimizer_for(
    module_type,
) -> tp.Callable[[tp.Type[GeneralizedOptimizer]], tp.Any]:
    def _lambda(generalized_module_type) -> tp.Any:
        if module_type in REGISTRY:
            raise ModuleExists(
                f"Type {module_type} already registered with {REGISTRY[module_type]}"
            )

        REGISTRY[module_type] = generalized_module_type

        return generalized_module_type

    return _lambda


def generalize_optimizer(optimizer: tp.Any) -> GeneralizedOptimizer:

    if isinstance(optimizer, GeneralizedOptimizer):
        return optimizer

    generalized_module_type: tp.Optional[tp.Type[GeneralizedOptimizer]] = None

    for module_type in REGISTRY:
        if isinstance(optimizer, module_type):
            generalized_module_type = REGISTRY[module_type]

    if generalized_module_type is None:
        raise ValueError(f"No GeneralizedOptimizer found for {optimizer}.")

    return generalized_module_type(optimizer)
