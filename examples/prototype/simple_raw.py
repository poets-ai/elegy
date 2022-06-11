import functools
import typing as tp
from dataclasses import dataclass
from functools import partial
from mimetypes import init

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax_metrics as jm
import matplotlib.pyplot as plt
import numpy as np
import optax
import typer
from attr import mutable
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


@dataclass
class CNN(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        # Normalize the input
        x = x.astype(jnp.float32) / 255.0

        # Block 1
        x = nn.Conv(32, [3, 3], strides=[2, 2])(x)
        x = nn.Dropout(0.05, deterministic=not training)(x)
        x = jax.nn.relu(x)

        # Block 2
        x = nn.Conv(64, [3, 3], strides=[2, 2])(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.Dropout(0.1, deterministic=not training)(x)
        x = jax.nn.relu(x)

        # Block 3
        x = nn.Conv(128, [3, 3], strides=[2, 2])(x)

        # Global average pooling
        x = x.mean(axis=(1, 2))

        # Classification layer
        x = nn.Dense(10)(x)

        return x


Batch = tp.Mapping[str, np.ndarray]
Metric = jm.metrics.Accuracy
Logs = tp.Mapping[str, jnp.ndarray]
Variables = FrozenDict[str, tp.Mapping[str, tp.Any]]
np.random.seed(420)

C = tp.TypeVar("C", bound="tp.Callable")


@partial(jax.jit, static_argnums=(2, 3))
def init_step(
    key: jnp.ndarray,
    batch: tp.Tuple[jnp.ndarray, jnp.ndarray],
    init_fn: tp.Callable[..., Variables],
    optimizer: optax.GradientTransformation,
) -> tp.Tuple[Variables, optax.OptState]:
    inputs, labels = batch

    variables = init_fn(key, inputs, training=False)
    opt_state = optimizer.init(variables["params"])

    return variables, opt_state


def loss_fn(
    params: tp.Mapping[str, jnp.ndarray],
    variables: Variables,
    key: tp.Optional[jnp.ndarray],
    inputs: jnp.ndarray,
    labels: jnp.ndarray,
    apply_fn: tp.Callable[..., tp.Tuple[jnp.ndarray, Variables]],
) -> tp.Tuple[jnp.ndarray, Variables]:
    variables = variables.copy({"params": params})

    preds, updates = apply_fn(
        variables,
        inputs,
        training=True,
        rngs={"dropout": key},
        mutable=["batch_stats"],
    )
    variables = variables.copy(updates)

    loss = optax.softmax_cross_entropy(
        preds,
        jax.nn.one_hot(labels, preds.shape[-1]),
    ).mean()

    return loss, variables


@partial(jax.jit)
def train_step(
    key: jnp.ndarray,
    batch: tp.Tuple[jnp.ndarray, jnp.ndarray],
    variables: Variables,
    opt_state: optax.OptState,
    apply_fn: tp.Callable[..., tp.Tuple[jnp.ndarray, Variables]],
    optimizer: optax.GradientTransformation,
) -> tp.Tuple[Logs, Variables, optax.OptState]:
    inputs, labels = batch

    params = variables["params"]
    loss_key, key = jax.random.split(key)

    (loss, variables), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, variables, loss_key, inputs, labels, apply_fn
    )

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    variables = variables.copy({"params": params})

    logs = {"loss": loss}

    return logs, variables, opt_state


X_train, y_train, X_test, y_test = get_data()

key = jax.random.PRNGKey(0)
epochs = 10
batch_size = 32
step_per_epoch = len(X_train) // batch_size

module = CNN()
optimizer = optax.adam(1e-3)

variables, opt_state = init_step(
    key, (X_train[:10], y_train[:10]), module.init, optimizer
)


for epoch in range(epochs):
    logs = {}

    for step in range(step_per_epoch):
        key, step_key = jax.random.split(key)
        idx = np.random.randint(0, len(X_train), batch_size)
        batch = (X_train[idx], y_train[idx])

        logs, variables, opt_state = train_step(
            step_key, batch, variables, opt_state, module.apply, optimizer
        )

    print(f"Epoch {epoch}: {logs}")
