import typing as tp
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import treex as tx
from einop import einop

import elegy.modules.core_module as cm


@dataclass(unsafe_hash=True)
class Eager(cm.DistributedStrategy):
    def init_step_fn(self, model: cm.M) -> cm.InitStep[cm.M]:
        return model.__class__.init_step

    def pred_step_fn(self, model: cm.M) -> cm.PredStep[cm.M]:
        return model.__class__.predict_step

    def test_step_fn(self, model: cm.M) -> cm.TestStep[cm.M]:
        return model.__class__.test_step

    def train_step_fn(self, model: cm.M) -> cm.TrainStep[cm.M]:
        return model.__class__.train_step


@dataclass(unsafe_hash=True)
class JIT(cm.DistributedStrategy):
    # donate 'model' memory buffer since we return an updated model
    def init_step_fn(self, model: cm.M) -> cm.InitStep[cm.M]:
        return jax.jit(model.__class__.init_step, donate_argnums=0)

    def pred_step_fn(self, model: cm.M) -> cm.PredStep[cm.M]:
        return jax.jit(model.__class__.predict_step, donate_argnums=0)

    def test_step_fn(self, model: cm.M) -> cm.TestStep[cm.M]:
        return jax.jit(model.__class__.test_step, donate_argnums=0)

    def train_step_fn(self, model: cm.M) -> cm.TrainStep[cm.M]:
        return jax.jit(model.__class__.train_step, donate_argnums=0)


@dataclass(unsafe_hash=True)
class DataParallel(cm.DistributedStrategy):
    def from_local(self, model: cm.M) -> cm.M:
        # device_idxs used to inform pmap about the number of devices
        device_idxs = jnp.arange(jax.device_count())
        model = jax.pmap(
            lambda idx, model: model,
            in_axes=(0, None),
            out_axes=0,
        )(device_idxs, model)

        return model

    def to_local(self, model: cm.M) -> cm.M:
        return jax.tree_map(lambda x: x[0], model)

    def lift_data(self, data: cm.A) -> cm.A:
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

    def handle_post_init(self, model: cm.M) -> cm.M:
        return model.map(
            lambda key: jax.random.fold_in(key, jax.lax.axis_index("device")),
            tx.Rng,
            inplace=True,
        )

    def handle_metrics(
        self,
        metrics: cm.Me,
    ) -> cm.Me:
        metrics = jax.lax.stop_gradient(metrics)
        metrics = jax.lax.psum(metrics, axis_name="device")
        return metrics

    def handle_model_and_grads(
        self,
        model: cm.M,
        grads: cm.M,
    ) -> tp.Tuple[cm.M, cm.M]:
        grads = jax.lax.pmean(grads, axis_name="device")
        model = model.map(lambda x: jax.lax.pmean(x, axis_name="device"), tx.BatchStat)

        return model, grads

    def init_step_fn(self, model: cm.M) -> cm.InitStep[cm.M]:
        return jax.pmap(
            model.__class__.init_step,
            axis_name="device",
            donate_argnums=0,
        )

    def pred_step_fn(self, model: cm.M) -> cm.PredStep[cm.M]:
        return jax.pmap(
            model.__class__.predict_step,
            axis_name="device",
            donate_argnums=0,
        )

    def test_step_fn(self, model: cm.M) -> cm.TestStep[cm.M]:
        return jax.pmap(
            model.__class__.test_step,
            axis_name="device",
            out_axes=(0, None, 0),  # None = logs not replicated
            donate_argnums=0,
        )

    def train_step_fn(self, model: cm.M) -> cm.TrainStep[cm.M]:
        return jax.pmap(
            model.__class__.train_step,
            axis_name="device",
            out_axes=(None, 0),  # None = logs not replicated
            donate_argnums=0,
        )
