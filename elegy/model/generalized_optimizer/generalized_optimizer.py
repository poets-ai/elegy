import typing as tp
from abc import ABC, abstractmethod

from elegy import utils
from elegy.types import OutputStates, RNGSeq
import typing as tp

REGISTRY: tp.Dict[tp.Type, tp.Type["GeneralizedOptimizer"]] = {}

NetParams = tp.Any
OptimizerStates = tp.Any
Grads = tp.Any


class ModuleExists(Exception):
    pass


class GeneralizedOptimizer(ABC):
    @abstractmethod
    def __init__(self, optimizer: tp.Any):
        ...

    @abstractmethod
    def init(self, rng: RNGSeq, net_params: NetParams) -> OptimizerStates:
        ...

    @abstractmethod
    def apply(
        self,
        net_params: NetParams,
        grads: Grads,
        optimizer_states: OptimizerStates,
        rng: RNGSeq,
    ) -> tp.Tuple[NetParams, OptimizerStates]:
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