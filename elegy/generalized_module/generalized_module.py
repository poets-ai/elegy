import typing as tp
from abc import ABC, abstractmethod

from elegy import types, utils

REGISTRY: tp.Dict[tp.Type, tp.Type["GeneralizedModule"]] = {}


class ModuleExists(Exception):
    pass


class GeneralizedModule(ABC):
    @abstractmethod
    def __init__(self, module: tp.Any):
        ...

    @abstractmethod
    def init(self, rng: types.RNGSeq) -> tp.Callable[..., types.OutputStates]:
        ...

    @abstractmethod
    def apply(
        self,
        params: tp.Any,
        states: tp.Any,
        training: bool,
        rng: types.RNGSeq,
    ) -> tp.Callable[..., types.OutputStates]:
        ...

    def get_summary_params(
        self,
        path: types.Path,
        module: tp.Any,
        value: tp.Any,
        include_submodules: bool,
        net_params: types.NetParams,
        net_states: types.NetStates,
    ) -> tp.Tuple[tp.Optional[types.Pytree], tp.Optional[types.Pytree]]:
        return None, None

    def update(
        self,
        params: tp.Optional[types.ModuleParams],
        states: tp.Optional[types.ModuleStates],
    ) -> tp.Tuple[tp.Optional[types.ModuleParams], tp.Optional[types.ModuleStates]]:
        return params, states


class CallableModule(GeneralizedModule):
    def __init__(self, f: tp.Callable):
        self.f = f

    def init(self, rng: types.RNGSeq) -> tp.Callable[..., types.OutputStates]:
        def lambda_(*args, **kwargs) -> types.OutputStates:

            output = utils.inject_dependencies(self.f)(*args, **kwargs)

            if isinstance(output, types.OutputStates):
                return output
            else:
                return types.OutputStates(
                    preds=output,
                    params=types.UNINITIALIZED,
                    states=types.UNINITIALIZED,
                )

        return lambda_

    def apply(
        self,
        params: tp.Any,
        states: tp.Any,
        training: bool,
        rng: types.RNGSeq,
    ) -> tp.Callable[..., types.OutputStates]:
        def lambda_(*args, **kwargs) -> types.OutputStates:

            output = utils.inject_dependencies(self.f)(*args, **kwargs)

            if isinstance(output, types.OutputStates):
                return output
            else:
                return types.OutputStates(
                    preds=output,
                    params=types.UNINITIALIZED,
                    states=types.UNINITIALIZED,
                )

        return lambda_


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


def is_generalizable(module: tp.Any, accept_callable: bool = False) -> bool:

    if isinstance(module, GeneralizedModule):
        return True

    for module_type in REGISTRY:
        if isinstance(module, module_type):
            return True

    if accept_callable and isinstance(module, tp.Callable):
        return True

    return False
