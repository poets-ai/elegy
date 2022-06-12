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
from responses import target

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


class CNNModule(eg.CoreModule):
    module: CNN = eg.static_field()
    optimizer: optax.GradientTransformation = eg.static_field()

    def __init__(
        self,
        module: CNN,
        optimizer: optax.GradientTransformation,
        *,
        initialized: bool = False,
    ) -> None:
        super().__init__(initialized=initialized)
        self.module = module
        self.optimizer = optimizer
        self.metrics = eg.LossesAndMetrics(
            losses=eg.losses.Crossentropy(),
            metrics=eg.metrics.Accuracy(),
        )
        self.key: tp.Optional[jnp.ndarray] = None
        self.params: tp.Optional[Variables] = None
        self.batch_stats: tp.Optional[Variables] = None
        self.opt_state: tp.Optional[tp.Any] = None

    @jax.jit
    def init_step(
        self: M,
        key: jnp.ndarray,
        batch: tp.Tuple[jnp.ndarray, jnp.ndarray],
    ) -> M:
        inputs, labels = batch
        init_key, key = jax.random.split(key)
        variables = self.module.init(init_key, inputs, training=False)
        return self.replace(
            key=key,
            params=variables["params"],
            batch_stats=variables["batch_stats"],
            opt_state=self.optimizer.init(variables["params"]),
            metrics=self.metrics.init(),
        )

    @jax.jit
    def reset_step(self: M) -> M:
        return self.replace(metrics=self.metrics.reset())

    def loss_fn(
        self: M,
        params: Variables,
        forward: tp.Callable[[M, Variables, jnp.ndarray], tp.Tuple[jnp.ndarray, M]],
        inputs: jnp.ndarray,
        labels: jnp.ndarray,
    ) -> tp.Tuple[jnp.ndarray, M]:

        preds, self = forward(self, params, inputs)
        loss, metrics = self.metrics.loss_and_update(preds=preds, target=labels)
        return loss, self.replace(metrics=metrics)

    @jax.jit
    def train_step(
        self: M,
        batch: tp.Tuple[jnp.ndarray, jnp.ndarray],
        batch_idx: int,
        epoch_idx: int,
    ) -> tp.Tuple[Logs, M]:
        inputs, labels = batch

        def forward(
            self: M, params: Variables, inputs: jnp.ndarray
        ) -> tp.Tuple[jnp.ndarray, M]:
            dropout_key, key = jax.random.split(self.key)
            preds, updates = self.module.apply(
                {"params": params, "batch_stats": self.batch_stats},
                inputs,
                training=True,
                rngs={"dropout": dropout_key},
                mutable=["batch_stats"],
            )
            return preds, self.replace(key=key, batch_stats=updates["batch_stats"])

        (loss, self), grads = jax.value_and_grad(self.loss_fn, has_aux=True)(
            self.params, forward, inputs, labels
        )
        updates, opt_state = self.optimizer.update(grads, self.opt_state, self.params)
        params = optax.apply_updates(self.params, updates)
        logs = self.metrics.compute_logs()
        return logs, self.replace(params=params, opt_state=opt_state)

    @jax.jit
    def test_step(
        self: M,
        batch: tp.Tuple[jnp.ndarray, jnp.ndarray],
        batch_idx: int,
    ) -> tp.Tuple[Logs, M]:
        inputs, labels = batch

        def forward(
            self: M, params: Variables, inputs: jnp.ndarray
        ) -> tp.Tuple[jnp.ndarray, M]:
            preds = self.module.apply(
                {"params": params, "batch_stats": self.batch_stats},
                inputs,
                training=False,
            )
            return preds, self

        loss, self = self.loss_fn(self.params, forward, inputs, labels)
        logs = self.metrics.compute_logs()
        return logs, self

    @jax.jit
    def predict_step(
        self: M, inputs: jnp.ndarray, batch_idx: int
    ) -> tp.Tuple[jnp.ndarray, M]:
        preds = self.module.apply(
            {"params": self.params, "batch_stats": self.batch_stats},
            inputs,
            training=False,
        )
        preds = jnp.argmax(preds, axis=-1)
        return preds, self


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
    callbacks=[eg.callbacks.TensorBoard("summaries/core/cnn_full")],
)
eg.utils.plot_history(history)

print(trainer.evaluate(X_test, y_test))

# get random samples
idxs = np.random.randint(0, len(X_test), size=(9,))
x_sample = X_test[idxs]

# get predictions
y_pred = trainer.predict(x_sample)

# plot results
figure = plt.figure(figsize=(12, 12))
for i in range(3):
    for j in range(3):
        k = 3 * i + j
        plt.subplot(3, 3, k + 1)

        plt.title(f"{y_pred[k]}")
        plt.imshow(x_sample[k], cmap="gray")

plt.show()
