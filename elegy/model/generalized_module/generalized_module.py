import typing as tp
from abc import ABC, abstractmethod

from elegy import utils
from elegy.types import OutputStates
import typing as tp

REGISTRY: tp.Dict[tp.Type, "GeneralizedModule"] = {}


class ModuleExists(Exception):
    pass


def register_module(module_type, generalized_module_type: "GeneralizedModule"):

    if module_type in REGISTRY:
        raise ModuleExists(
            f"Type {module_type} already registered with {REGISTRY[module_type]}"
        )

    REGISTRY[module_type] = generalized_module_type


def generalize(module: tp.Any) -> "GeneralizedModule":

    if isinstance(module, GeneralizedModule):
        return module

    generalized_module_type: tp.Optional[GeneralizedModule] = None

    for module_type in REGISTRY:
        if isinstance(module, module_type):
            generalized_module_type = REGISTRY[module_type]

    if generalized_module_type is None:
        raise ValueError(f"No GeneralizedModule found for {module}.")

    return generalized_module_type.new(module)


class GeneralizedModule(ABC):
    @classmethod
    @abstractmethod
    def new(cls, module: tp.Any) -> "GeneralizedModule":
        ...

    @abstractmethod
    def init(
        self, rng: utils.RNGSeq, args: tp.Tuple, kwargs: tp.Dict[str, tp.Any]
    ) -> OutputStates:
        ...

    @abstractmethod
    def apply(
        self,
        params: tp.Any,
        states: tp.Any,
        rng: utils.RNGSeq,
        args: tp.Tuple,
        kwargs: tp.Dict[str, tp.Any],
    ) -> OutputStates:
        ...
