import functools
import typing as tp
from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax_metrics as jm
import treex as tx
from einop import einop

import elegy as eg

StrategyConstructor = tp.Callable[[], "Strategy"]
M = tp.TypeVar("M", bound="eg.ManagedModule")
ME = tp.TypeVar("ME", bound="jm.Metric")
A = tp.TypeVar("A")

_REGISTRY: tp.Dict[str, StrategyConstructor] = {}


@tp.overload
def register_strategy(
    name: str,
) -> tp.Callable[[StrategyConstructor], StrategyConstructor]:
    ...


@tp.overload
def register_strategy(
    name: str,
    *,
    constructor: StrategyConstructor,
) -> None:
    ...


def register_strategy(
    name: str,
    *,
    constructor: tp.Optional[StrategyConstructor] = None,
) -> tp.Optional[tp.Callable[[StrategyConstructor], StrategyConstructor]]:
    """
    Register a strategy class.
    """

    def _register(constructor: StrategyConstructor):
        if name in _REGISTRY:
            raise ValueError(f"Strategy {name} already registered")

        _REGISTRY[name] = constructor

    if constructor is None:

        def decorator(
            constructor: StrategyConstructor,
        ) -> StrategyConstructor:
            _register(constructor)
            return constructor

        return decorator
    else:
        _register(constructor)


def get_strategy(name: str) -> "Strategy":
    """
    Get a strategy class.
    """
    if name not in _REGISTRY:
        raise ValueError(f"Strategy {name} not registered")

    return _REGISTRY[name]()


class Strategy(ABC):
    def from_local(self, module: M) -> M:
        return module

    def to_local(self, module: M) -> M:
        return module

    def lift_data(self, data: A) -> A:
        return data

    def lift_key(self, key: jnp.ndarray) -> jnp.ndarray:
        return key

    def lift_batch_size(self, batch_size: int) -> int:
        return batch_size

    def handle_post_init(self, module: M) -> M:
        return module

    def handle_metrics(
        self,
        metrics: ME,
    ) -> ME:
        return metrics

    def handle_grads(
        self,
        grads: tp.Any,
    ) -> tp.Any:
        return grads

    def handle_batch_stats(
        self,
        batch_stats: tp.Any,
    ) -> tp.Any:
        return batch_stats

    @abstractmethod
    def init_step_fn(
        self, module: "eg.modules.ManagedModule"
    ) -> "eg.modules.InitStep[eg.modules.ManagedModule]":
        ...

    @abstractmethod
    def reset_step_fn(
        self, module: "eg.modules.ManagedModule"
    ) -> "eg.modules.ResetStep[eg.modules.ManagedModule]":
        ...

    @abstractmethod
    def predict_step_fn(
        self, module: "eg.modules.ManagedModule"
    ) -> "eg.modules.PredStep[eg.modules.ManagedModule]":
        ...

    @abstractmethod
    def test_step_fn(
        self, module: "eg.modules.ManagedModule"
    ) -> "eg.modules.TestStep[eg.modules.ManagedModule]":
        ...

    @abstractmethod
    def train_step_fn(
        self, module: "eg.modules.ManagedModule"
    ) -> "eg.modules.TrainStep[eg.modules.ManagedModule]":
        ...

    # implement order methods, required so that Strategy can be
    # used as a key in a dict with JAX
    def __lt__(self, other):
        return self.__class__.__name__ < other.__class__.__name__

    def __le__(self, other):
        return self.__class__.__name__ <= other.__class__.__name__

    def __gt__(self, other):
        return self.__class__.__name__ > other.__class__.__name__

    def __ge__(self, other):
        return self.__class__.__name__ >= other.__class__.__name__


@register_strategy("eager")
@dataclass(unsafe_hash=True)
class Eager(Strategy):
    def init_step_fn(self, module: M) -> "eg.modules.InitStep[M]":
        return module.__class__._init_step_manager

    def reset_step_fn(self, module: M) -> "eg.modules.ResetStep[M]":
        return module.__class__._reset_step_manager

    def predict_step_fn(self, module: M) -> "eg.modules.PredStep[M]":
        return module.__class__._predict_step_manager

    def test_step_fn(self, module: M) -> "eg.modules.TestStep[M]":
        return module.__class__._test_step_manager

    def train_step_fn(self, module: M) -> "eg.modules.TrainStep[M]":
        return module.__class__._train_step_manager


@register_strategy("jit")
@dataclass(unsafe_hash=True)
class JIT(Strategy):
    donate_args: bool = False

    def init_step_fn(self, module: M) -> "eg.modules.InitStep[M]":
        return jax.jit(
            module.__class__._init_step_manager,
            donate_argnums=0 if self.donate_args else (),
        )

    def reset_step_fn(self, module: M) -> "eg.modules.ResetStep[M]":
        return jax.jit(
            module.__class__._reset_step_manager,
            donate_argnums=0 if self.donate_args else (),
        )

    def predict_step_fn(self, module: M) -> "eg.modules.PredStep[M]":
        return jax.jit(
            module.__class__._predict_step_manager,
            donate_argnums=0 if self.donate_args else (),
        )

    def test_step_fn(self, module: M) -> "eg.modules.TestStep[M]":
        return jax.jit(
            module.__class__._test_step_manager,
            donate_argnums=0 if self.donate_args else (),
        )

    def train_step_fn(self, module: M) -> "eg.modules.TrainStep[M]":
        return jax.jit(
            module.__class__._train_step_manager,
            donate_argnums=0 if self.donate_args else (),
        )


@register_strategy("data_parallel")
@dataclass(unsafe_hash=True)
class DataParallel(Strategy):
    axis_name: str = "device"
    donate_args: bool = False

    def from_local(self, module: M) -> M:
        # device_idxs used to inform pmap about the number of devices
        device_idxs = jnp.arange(jax.device_count())
        module = jax.pmap(
            lambda idx, module: module,
            in_axes=(0, None),
            out_axes=0,
        )(device_idxs, module)

        if module.key is not None:
            # this is the same a handle_post_init, maybe use that?
            # give unique key to each device
            device_idx = jax.lax.axis_index(self.axis_name)
            module = module.replace(
                key=jax.random.fold_in(module.key, device_idx),
            )

        return module

    def to_local(self, module: M) -> M:
        module = jax.tree_map(lambda x: x[0], module)
        return module

    def lift_data(self, data: A) -> A:
        data = jax.tree_map(
            lambda x: einop(
                x,
                "(device batch) ... -> device batch ...",
                device=jax.device_count(),
            ),
            data,
        )
        return data

    def lift_key(self, key: jnp.ndarray) -> jnp.ndarray:
        key = einop(
            key,
            "... -> device ...",
            device=jax.device_count(),
        )
        return key

    def lift_batch_size(self, batch_size: int) -> int:
        return batch_size * jax.device_count()

    def handle_post_init(self, module: M) -> M:

        # give unique key to each device
        device_idx = jax.lax.axis_index(self.axis_name)
        return module.replace(
            key=jax.random.fold_in(module.key, device_idx),
        )

    def handle_metrics(
        self,
        metrics: ME,
    ) -> ME:
        metrics = jax.lax.stop_gradient(metrics)

        metrics = jax.lax.all_gather(metrics, axis_name=self.axis_name)
        metrics = metrics.aggregate()

        return metrics

    def handle_grads(self, grads: A) -> A:
        return jax.lax.pmean(grads, axis_name=self.axis_name)

    def handle_batch_stats(self, batch_stats: A) -> A:
        return jax.lax.pmean(batch_stats, axis_name=self.axis_name)

    def init_step_fn(self, module: M) -> "eg.modules.InitStep[M]":
        return jax.pmap(
            module.__class__._init_step_manager,
            axis_name=self.axis_name,
            donate_argnums=0 if self.donate_args else (),
        )

    def reset_step_fn(self, module: M) -> "eg.modules.ResetStep[M]":
        return jax.pmap(
            module.__class__._reset_step_manager,
            axis_name=self.axis_name,
            donate_argnums=0 if self.donate_args else (),
        )

    def predict_step_fn(self, module: M) -> "eg.modules.PredStep[M]":
        return jax.pmap(
            module.__class__._predict_step_manager,
            axis_name=self.axis_name,
            in_axes=(0, 0, None),  # None = batch_idx not replicaed
            donate_argnums=0 if self.donate_args else (),
        )

    def test_step_fn(self, module: M) -> "eg.modules.TestStep[M]":
        return jax.pmap(
            module.__class__._test_step_manager,
            axis_name=self.axis_name,
            in_axes=(0, 0, None),  # None = batch_idx not replicaed
            out_axes=(None, 0),  # None = logs not replicated
            donate_argnums=0 if self.donate_args else (),
        )

    def train_step_fn(self, module: M) -> "eg.modules.TrainStep[M]":
        return jax.pmap(
            module.__class__._train_step_manager,
            axis_name=self.axis_name,
            in_axes=(0, 0, None, None),  # None = batch_idx and epoch_idx not replicaed
            out_axes=(None, 0),  # None = logs not replicated
            donate_argnums=0 if self.donate_args else (),
        )
