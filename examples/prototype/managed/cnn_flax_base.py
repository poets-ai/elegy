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
from attr import mutable
from datasets.load import load_dataset
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

import elegy as eg


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

M = tp.TypeVar("M", bound="CNNModule")
C = tp.TypeVar("C", bound="tp.Callable")


class CNNModule(eg.ManagedModule):
    module: Module = eg.static_field()

    def __init__(
        self,
        module: Module,
    ) -> None:
        super().__init__()
        self.module = module

    def managed_init_step(
        self: M,
        key: jnp.ndarray,
        batch: tp.Tuple[jnp.ndarray, jnp.ndarray],
    ) -> M:
        inputs, labels = batch

        variables = self.module.init(key, inputs, training=False)

        return self.replace(
            params=variables["params"],
            batch_stats=variables["batch_stats"],
        )

    def loss_fn(
        self: M,
        key: tp.Optional[jnp.ndarray],
        batch: tp.Tuple[jnp.ndarray, jnp.ndarray],
        training: bool,
    ) -> tp.Tuple[eg.types.Loss, M]:
        assert self.params is not None

        inputs, labels = batch

        outputs = self.module.apply(
            {"params": self.params, "batch_stats": self.batch_stats},
            inputs,
            training=training,
            rngs={"dropout": key} if training else {},
            mutable=["batch_stats"] if training else False,
        )

        preds: jnp.ndarray
        if training:
            preds, updates = outputs
            self = self.replace(batch_stats=updates["batch_stats"])
        else:
            preds = outputs

        oh_labels = jax.nn.one_hot(labels, preds.shape[-1])
        loss = jnp.mean(optax.softmax_cross_entropy(preds, oh_labels))
        accuracy = jnp.mean(jnp.argmax(preds, axis=-1) == labels)

        self = self.log("accuracy", accuracy)

        return loss, self

    def managed_train_step(
        self: M,
        key: jnp.ndarray,
        batch: tp.Tuple[jnp.ndarray, jnp.ndarray],
        batch_idx: jnp.ndarray,
        epoch_idx: jnp.ndarray,
    ) -> tp.Tuple[eg.types.Loss, M]:
        print("JITTING")
        return self.loss_fn(key, batch, training=True)

    def managed_test_step(
        self: M, key: jnp.ndarray, batch: tp.Any, batch_idx: jnp.ndarray
    ) -> M:
        loss, self = self.loss_fn(key, batch, training=False)
        return self

    def managed_predict_step(
        self: M, key: jnp.ndarray, batch: jnp.ndarray, batch_idx: jnp.ndarray
    ) -> tp.Tuple[eg.types.Outputs, M]:
        assert self.params is not None and self.batch_stats is not None
        inputs = batch

        outputs = self.module.apply(
            {"params": self.params, "batch_stats": self.batch_stats},
            inputs,
            training=False,
            mutable=False,
            rngs={},
        )
        outputs = jnp.argmax(outputs, axis=1)
        return outputs, self

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

    print("X_train:", X_train.shape, X_train.dtype)
    print("X_test:", X_test.shape, X_test.dtype)

    model = eg.Trainer(
        module=CNNModule(CNN()), optimizer=optax.adamw(1e-3), strategy="jit", seed=seed
    )

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

    model.set_strategy("jit")

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
