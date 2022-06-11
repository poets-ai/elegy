import typing as tp

import optax
import treeo as to

import elegy.pytree as pytree_m
from elegy import types
from elegy.modules.high_level.flax_module import FlaxMixin
from elegy.modules.managed.managed_module import ManagedModule
from elegy.optimizer import Optimizer

try:
    import flax.linen as nn
    from flax.core.frozen_dict import FrozenDict

except (ImportError, ModuleNotFoundError):
    raise types.DependencyUnavailable("Flax not avialable.")

F = tp.TypeVar("F", bound="nn.module.Module")
Variables = tp.Mapping[str, tp.Mapping[str, tp.Any]]


class ManagedFlaxModule(tp.Generic[F], FlaxMixin, ManagedModule):
    _module: tp.Callable[[], F] = pytree_m.static_field()

    def __init__(
        self,
        module: F,
        *,
        variables: tp.Optional[Variables] = None,
        optimizer: tp.Optional[
            tp.Union[optax.GradientTransformation, Optimizer]
        ] = None,
        initialized: bool = False,
        strategy: tp.Optional[tp.Union[str, "eg.Strategy"]] = None,
    ) -> None:
        super().__init__(
            optimizer=optimizer, initialized=initialized, strategy=strategy
        )
        self._module = lambda: module
        self.variables = FrozenDict(variables) if variables is not None else None

    @property
    def module(self) -> F:
        return self._module()
