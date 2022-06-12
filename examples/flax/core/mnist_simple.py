import functools
import typing as tp
from dataclasses import dataclass
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax_metrics as jm
import matplotlib.pyplot as plt
import numpy as np
import optax
from datasets.load import load_dataset
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

import elegy as eg
from elegy.extras.flax_module import ModuleState


def get_data():
    dataset = load_dataset("mnist")
    dataset.set_format("np")
    X_train = np.stack(dataset["train"]["image"])[..., None]
    y_train = dataset["train"]["label"]
    X_test = np.stack(dataset["test"]["image"])[..., None]
    y_test = dataset["test"]["label"]
    return X_train, y_train, X_test, y_test


Batch = tp.Mapping[str, np.ndarray]
Metric = jm.metrics.Accuracy
Logs = tp.Mapping[str, jnp.ndarray]
Variables = FrozenDict[str, tp.Mapping[str, tp.Any]]
np.random.seed(420)

M = tp.TypeVar("M", bound="CNNModule")
C = tp.TypeVar("C", bound="tp.Callable")


@dataclass
class CNN(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        x = x.astype(jnp.float32) / 255.0
        x = nn.Conv(32, [3, 3], strides=[2, 2])(x)
        x = nn.Dropout(0.05, deterministic=not training)(x)
        x = jax.nn.relu(x)
        x = nn.Conv(64, [3, 3], strides=[2, 2])(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.Dropout(0.1, deterministic=not training)(x)
        x = jax.nn.relu(x)
        x = nn.Conv(128, [3, 3], strides=[2, 2])(x)
        x = x.mean(axis=(1, 2))
        x = nn.Dense(10)(x)
        return x


@dataclass
class CNNModule(eg.CoreModule):
    module: CNN = eg.static_field()
    optimizer: optax.GradientTransformation = eg.static_field()
    key: tp.Optional[jnp.ndarray] = None
    params: tp.Optional[Variables] = None
    batch_stats: tp.Optional[Variables] = None
    opt_state: tp.Optional[tp.Any] = None

    @jax.jit
    def init_step(
        self: M,
        key: jnp.ndarray,
        batch: tp.Tuple[jnp.ndarray, jnp.ndarray],
    ) -> M:
        inputs, labels = batch
        init_key, key = jax.random.split(key)
        variables = self.module.init(init_key, inputs, training=False)
        opt_state = self.optimizer.init(variables["params"])
        return self.replace(
            key=key,
            params=variables["params"],
            batch_stats=variables["batch_stats"],
            opt_state=opt_state,
        )

    def loss_fn(
        self: "CNNModule",
        params: tp.Mapping[str, jnp.ndarray],
        key: tp.Optional[jnp.ndarray],
        inputs: jnp.ndarray,
        labels: jnp.ndarray,
    ) -> tp.Tuple[jnp.ndarray, "CNNModule"]:
        preds, updates = self.module.apply(
            {"params": params, "batch_stats": self.batch_stats},
            inputs,
            training=True,
            rngs={"dropout": key},
            mutable=["batch_stats"],
        )
        loss = optax.softmax_cross_entropy(
            preds, jax.nn.one_hot(labels, preds.shape[-1])
        ).mean()
        return loss, self.replace(batch_stats=updates["batch_stats"])

    @jax.jit
    def train_step(
        self: M,
        batch: tp.Tuple[jnp.ndarray, jnp.ndarray],
        batch_idx: int,
        epoch_idx: int,
    ) -> tp.Tuple[Logs, M]:
        inputs, labels = batch
        loss_key, key = jax.random.split(self.key)
        (loss, self), grads = jax.value_and_grad(self.loss_fn, has_aux=True)(
            self.params, loss_key, inputs, labels
        )
        updates, opt_state = self.optimizer.update(grads, self.opt_state, self.params)
        params = optax.apply_updates(self.params, updates)
        logs = {"loss": loss}
        return logs, self.replace(key=key, params=params, opt_state=opt_state)


X_train, y_train, X_test, y_test = get_data()

trainer = eg.Trainer(
    CNNModule(module=CNN(), optimizer=optax.adamw(1e-3)),
)

history = trainer.fit(
    inputs=X_train,
    labels=y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test),
)
