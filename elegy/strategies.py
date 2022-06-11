import typing as tp
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import jax_metrics as jm
from einop import einop

import elegy as eg

StrategyConstructor = tp.Callable[[], "Strategy"]
M = tp.TypeVar("M", bound="eg.ManagedModule")
ME = tp.TypeVar("ME", bound="jm.Metric")
A = tp.TypeVar("A")

_REGISTRY: tp.Dict[str, StrategyConstructor] = {}


# ----------------------------------------------------------------------------
# utils
# ----------------------------------------------------------------------------


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


# ----------------------------------------------------------------------------
# register strategies
# ----------------------------------------------------------------------------

register_strategy(
    name="eager",
    constructor=lambda: Eager(),
)
register_strategy(
    name="jit",
    constructor=lambda: JIT(donate_args=False),
)
register_strategy(
    name="jit_donate",
    constructor=lambda: JIT(donate_args=True),
)
register_strategy(
    name="data_parallel",
    constructor=lambda: DataParallel(donate_args=False),
)
register_strategy(
    name="data_parallel_donate",
    constructor=lambda: DataParallel(donate_args=True),
)

# ----------------------------------------------------------------------------
# Strategy
# ----------------------------------------------------------------------------
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

    def lower_outputs(self, outputs: A) -> A:
        return outputs

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
        self, module: "eg.ManagedModule"
    ) -> "eg.modules.InitStep[eg.ManagedModule]":
        ...

    @abstractmethod
    def reset_step_fn(
        self, module: "eg.ManagedModule"
    ) -> "eg.modules.ResetStep[eg.ManagedModule]":
        ...

    @abstractmethod
    def predict_step_fn(
        self, module: "eg.ManagedModule"
    ) -> "eg.modules.PredStep[eg.ManagedModule]":
        ...

    @abstractmethod
    def test_step_fn(
        self, module: "eg.ManagedModule"
    ) -> "eg.modules.TestStep[eg.ManagedModule]":
        ...

    @abstractmethod
    def train_step_fn(
        self, module: "eg.ManagedModule"
    ) -> "eg.modules.TrainStep[eg.ManagedModule]":
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


# ----------------------------------------------------------------------------
# EAGER
# ----------------------------------------------------------------------------


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


# ----------------------------------------------------------------------------
# JIT
# ----------------------------------------------------------------------------


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


# ----------------------------------------------------------------------------
# Data Parallel
# ----------------------------------------------------------------------------


@partial(jax.pmap, in_axes=(None, 0), out_axes=0)
def _lift_module(module: M, device_idx: jnp.ndarray) -> M:
    return module.replace(
        key=jax.random.fold_in(module.key, device_idx),
    )


@dataclass(unsafe_hash=True)
class DataParallel(Strategy):
    axis_name: str = "device"
    donate_args: bool = False

    def from_local(self, module: M) -> M:
        # device_idxs used to inform pmap about the number of devices
        device_idxs = jnp.arange(jax.device_count())
        module = _lift_module(module, device_idxs)

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

    def lower_outputs(self, outputs: A) -> A:
        outputs = jax.tree_map(
            lambda x: einop(x, "device batch ... -> (device batch) ..."),
            outputs,
        )
        return outputs

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
