import typing as tp

import optax
import treeo as to

import elegy.pytree as pytree_m
from elegy import types
from elegy.modules.managed.managed_module import ManagedModule
from elegy.optimizer import Optimizer

try:
    import flax.linen as nn
    from flax.core.frozen_dict import FrozenDict

except (ImportError, ModuleNotFoundError):
    raise types.DependencyUnavailable("Flax not avialable.")

F = nn.module.Module
Variables = tp.Mapping[str, tp.Mapping[str, tp.Any]]


class ManagedFlaxModule(ManagedModule):
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

        if variables is not None:
            variables = FrozenDict(variables)

            if "params" in variables:
                variables, self.params = variables.pop("params")
            if "batch_stats" in variables:
                variables, self.batch_stats = variables.pop("batch_stats")

        self.variables = variables

    @property
    def module(self) -> F:
        return self._module()
