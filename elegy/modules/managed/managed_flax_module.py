import typing as tp

import flax.linen as nn
import optax
import treex as tx
from flax.core.frozen_dict import FrozenDict

from elegy.modules.high_level.flax_module import FlaxMixin
from elegy.modules.managed.managed_module import ManagedModule

F = tp.TypeVar("F", bound="nn.module.Module")
Variables = tp.Mapping[str, tp.Mapping[str, tp.Any]]


class ManagedFlaxModule(tp.Generic[F], FlaxMixin, ManagedModule):
    _module: tx.Hashable[F]

    def __init__(
        self,
        module: F,
        *,
        variables: tp.Optional[Variables] = None,
        optimizer: tp.Optional[
            tp.Union[optax.GradientTransformation, tx.Optimizer]
        ] = None,
        initialized: bool = False,
        strategy: tp.Optional[tp.Union[str, "eg.Strategy"]] = None,
    ) -> None:
        super().__init__(
            optimizer=optimizer, initialized=initialized, strategy=strategy
        )
        self._module = tx.Hashable(module)
        self.variables = FrozenDict(variables) if variables is not None else None

    @property
    def module(self) -> F:
        return self._module.value
