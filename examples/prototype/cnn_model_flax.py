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
import typer
from datasets.load import load_dataset
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

import elegy as eg
from elegy.model.model_full import Model
from elegy.modules.core_module import CoreModule
from elegy.modules.flax_module import ModuleState


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
Module = CNN
Metric = jm.metrics.Accuracy
Logs = tp.Mapping[str, jnp.ndarray]
np.random.seed(420)

M = tp.TypeVar("M", bound="ElegyModule")
C = tp.TypeVar("C", bound="tp.Callable")


class ElegyModule(CoreModule):
    key: tp.Optional[jnp.ndarray] = eg.node()
    module: ModuleState[Module] = eg.node()
    optimizer: eg.Optimizer = eg.node()
    losses_and_metrics: jm.LossesAndMetrics = eg.node()

    def __init__(
        self,
        module: Module,
        optimizer: optax.GradientTransformation,
        losses: tp.Any,
        metrics: tp.Any,
    ) -> None:
        super().__init__()
        self.key = None
        self.losses_and_metrics = jm.LossesAndMetrics(
            losses=losses,
            metrics=metrics,
        )
        self.module = ModuleState(module)
        self.optimizer = eg.Optimizer(optimizer)

    @jax.jit
    def init_step(
        self: M,
        key: jnp.ndarray,
        batch: tp.Union[jnp.ndarray, tp.Tuple[jnp.ndarray, jnp.ndarray]],
    ) -> M:
        if isinstance(batch, tuple):
            inputs, _ = batch
        else:
            inputs = batch

        init_key, key = jax.random.split(key)
        module = self.module.init(init_key, inputs, training=False)
        optimizer = self.optimizer.init(module["params"])
        losses_and_metrics = self.losses_and_metrics.reset()

        return self.replace(
            key=key,
            module=module,
            optimizer=optimizer,
            losses_and_metrics=losses_and_metrics,
        )

    @jax.jit
    def reset_step(self: M) -> M:
        return self.replace(
            losses_and_metrics=self.losses_and_metrics.reset(),
        )

    def loss_fn(
        self: "ElegyModule",
        params: tp.Optional[tp.Mapping[str, jnp.ndarray]],
        key: tp.Optional[jnp.ndarray],
        inputs: jnp.ndarray,
        labels: jnp.ndarray,
        training: bool,
    ) -> tp.Tuple[jnp.ndarray, "ElegyModule"]:

        module = self.module

        if params is not None:
            module = module.update(params=params)

        preds, module = module.apply(
            key,
            inputs,
            training=training,
        )

        loss, losses_and_metrics = self.losses_and_metrics.loss_and_update(
            target=labels,
            preds=preds,
        )

        return loss, self.replace(
            module=module,
            losses_and_metrics=losses_and_metrics,
        )

    @jax.jit
    def train_step(
        self: M,
        batch: tp.Tuple[jnp.ndarray, jnp.ndarray],
        batch_idx: int,
        epoch_idx: int,
    ) -> tp.Tuple[Logs, M]:
        print("JITTTTING")
        assert self.key is not None

        inputs, labels = batch

        params = self.module["params"]
        loss_key, key = jax.random.split(self.key)

        grads, self = jax.grad(self.loss_fn, has_aux=True)(
            params, loss_key, inputs, labels, training=True
        )

        params, optimizer = self.optimizer.update(grads, params)
        module = self.module.update(params=params)

        logs = self.losses_and_metrics.compute_logs()

        return logs, self.replace(
            key=key,
            module=module,
            optimizer=optimizer,
        )

    @jax.jit
    def test_step(
        self: M,
        batch: tp.Tuple[jnp.ndarray, jnp.ndarray],
        batch_idx: int,
    ) -> tp.Tuple[Logs, M]:
        inputs, labels = batch

        loss, self = self.loss_fn(None, None, inputs, labels, training=False)

        logs = self.losses_and_metrics.compute_logs()

        return logs, self

    @jax.jit
    def predict_step(
        self: M,
        batch: jnp.ndarray,
        batch_idx: int,
    ) -> tp.Tuple[tp.Any, M]:
        inputs = batch
        outputs, module = self.module.apply(
            None,
            inputs,
            training=False,
        )
        outputs = jnp.argmax(outputs, axis=1)

        return outputs, self.replace(module=module)

    def tabulate(
        self,
        inputs,
        summary_depth: int = 2,
    ) -> str:
        return "TODO"


# define parameters
def main(
    epochs: int = 10,
    batch_size: int = 32,
    steps_per_epoch: tp.Optional[int] = None,
    seed: int = 420,
):

    # load data
    dataset = load_dataset("mnist")
    dataset.set_format("np")
    X_train = np.stack(dataset["train"]["image"])[..., None]
    y_train = dataset["train"]["label"]
    X_test = np.stack(dataset["test"]["image"])[..., None]
    y_test = dataset["test"]["label"]

    # define model
    module = ElegyModule(
        module=CNN(),
        optimizer=optax.adamw(1e-3),
        losses=jm.losses.Crossentropy(),
        metrics=jm.metrics.Accuracy(),
    )

    print("X_train:", X_train.shape, X_train.dtype)
    print("X_test:", X_test.shape, X_test.dtype)

    model = Model(module)

    history = model.fit(
        inputs=X_train,
        labels=y_train,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        shuffle=True,
        callbacks=[eg.callbacks.TensorBoard("summaries/prototype")],
    )

    eg.utils.plot_history(history)

    print(model.evaluate(X_test, y_test))

    # get random samples
    idxs = np.random.randint(0, 10000, size=(9,))
    x_sample = X_test[idxs]

    # get predictions
    y_pred = model.predict(x_sample)

    # plot results
    figure = plt.figure(figsize=(12, 12))
    for i in range(3):
        for j in range(3):
            k = 3 * i + j
            plt.subplot(3, 3, k + 1)

            plt.title(f"{y_pred[k]}")
            plt.imshow(x_sample[k], cmap="gray")

    plt.show()


if __name__ == "__main__":

    typer.run(main)
