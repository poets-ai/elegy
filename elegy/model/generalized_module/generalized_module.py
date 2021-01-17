import typing as tp
from abc import ABC, abstractmethod

from elegy import utils
from elegy.types import OutputStates
import typing as tp

REGISTRY: tp.Dict[tp.Type, tp.Type["GeneralizedModule"]] = {}


class ModuleExists(Exception):
    pass


class GeneralizedModule(ABC):
    @abstractmethod
    def __init__(self, module: tp.Any):
        ...

    @abstractmethod
    def init(self, rng: utils.RNGSeq) -> tp.Callable[..., OutputStates]:
        ...

    @abstractmethod
    def apply(
        self,
        params: tp.Any,
        states: tp.Any,
        rng: utils.RNGSeq,
    ) -> tp.Callable[..., OutputStates]:
        ...


class CallableModule(GeneralizedModule):
    def __init__(self, f: tp.Callable):
        self.f = f

    def init(self, rng: utils.RNGSeq) -> tp.Callable[..., OutputStates]:
        return lambda *args, **kwargs: OutputStates(
            preds=utils.inject_dependencies(self.f)(*args, **kwargs),
            params=None,
            states=None,
        )

    def apply(
        self,
        params: tp.Any,
        states: tp.Any,
        rng: utils.RNGSeq,
    ) -> tp.Callable[..., OutputStates]:
        return lambda *args, **kwargs: OutputStates(
            preds=utils.inject_dependencies(self.f)(*args, **kwargs),
            params=None,
            states=None,
        )


def register_module_for(
    module_type,
) -> tp.Callable[[tp.Type[GeneralizedModule]], tp.Any]:
    def wrapper(generalized_module_type: tp.Type[GeneralizedModule]) -> tp.Any:
        if module_type in REGISTRY:
            raise ModuleExists(
                f"Type {module_type} already registered with {REGISTRY[module_type]}"
            )

        REGISTRY[module_type] = generalized_module_type

        return generalized_module_type

    return wrapper


def generalize(
    module: tp.Any,
    callable_default: tp.Type[GeneralizedModule] = CallableModule,
) -> GeneralizedModule:

    if isinstance(module, GeneralizedModule):
        return module

    generalized_module_type: tp.Optional[tp.Type[GeneralizedModule]] = None

    for module_type in REGISTRY:
        if isinstance(module, module_type):
            generalized_module_type = REGISTRY[module_type]

    if generalized_module_type is None:
        if isinstance(module, tp.Callable):
            return callable_default(module)
        else:
            raise ValueError(f"No GeneralizedModule found for {module}.")

    return generalized_module_type(module)