from typing import Any, Generator, Mapping, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
import typer

import elegy

OptState = Any
Batch = Mapping[str, np.ndarray]
np.random.seed(42)
import logging

logging.basicConfig(level=logging.DEBUG)


def load_dataset(
    split: str, *, is_training: bool, batch_size: int, for_fit=False
) -> Generator[Batch, None, None]:
    """Loads the dataset as a generator of batches."""
    ds = tfds.load("mnist:3.*.*", split=split).cache()
    if is_training:
        ds = ds.repeat()
        ds = ds.shuffle(10 * batch_size, seed=0)
    if for_fit:
        ds = ds.map(lambda row: (row["image"], row["label"]))
    ds = ds.batch(batch_size)
    return tfds.as_numpy(ds)


def plot_history(history):
    n_plots = len(history.history.keys()) // 2
    plt.figure(figsize=(8, 12))
    for i, key in enumerate(list(history.history.keys())[:n_plots]):
        metric = history.history[key]
        val_metric = history.history[f"val_{key}"]

        plt.subplot(n_plots, 1, i + 1)
        plt.plot(metric, label=f"Training {key}")
        plt.plot(val_metric, label=f"Validation {key}")
        plt.legend(loc="lower right")
        plt.ylabel(key)
        # plt.ylim([min(plt.ylim()), 1])
        plt.title(f"Training and Validation {key}")
    plt.show()


import time
import tensorflow as tf


def main(debug: bool = False, eager: bool = False):

    if debug:
        import debugpy

        print("Waiting for debugger...")
        debugpy.listen(5678)
        debugpy.wait_for_client()

    # Make datasets.
    train = load_dataset("train", is_training=True, batch_size=64)
    # train_eval = load_dataset("train", is_training=False, batch_size=1000)
    test_eval = load_dataset("test", is_training=False, batch_size=1000)

    n_iters = 1000
    x = []
    y = []
    for _, ex in zip(range(n_iters), train):
        x.append(ex["image"])
        y.append(ex["label"])
    y = np.concatenate(y, axis=0)
    x = np.vstack(x)
    print(x.shape, y.shape)

    x_val = []
    y_val = []
    for _, ex in zip(range(2), test_eval):
        x_val.append(ex["image"])
        y_val.append(ex["label"])
    y_val = np.concatenate(y_val, axis=0)
    x_val = np.vstack(x_val)
    print(x_val.shape, y_val.shape)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(300, activation="relu"),
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    epochs = 10
    # Fit with datasets in memory
    # history = model.fit(
    #     x=x,
    #     y=y,
    #     epochs=epochs,
    #     batch_size=64,
    #     initial_epoch=epochs * 0,
    #     steps_per_epoch=100,
    #     validation_data=(x_val, y_val),
    #     shuffle=True,
    #     verbose=2,
    # )
    # plot_history(history)
    # # exit()

    # # Fit with validation from train
    # history = model.fit(
    #     x=x,
    #     y=y,
    #     epochs=2 * epochs,
    #     initial_epoch=epochs * 1,
    #     batch_size=64,
    #     steps_per_epoch=100,
    #     validation_split=0.2,
    #     shuffle=True,
    # )
    # exit()

    # Fit with generators
    x = load_dataset("train", is_training=True, batch_size=64, for_fit=True)
    validation = load_dataset("test", is_training=False, batch_size=64, for_fit=True)

    history = model.fit(
        x,
        epochs=3 * epochs,
        initial_epoch=epochs * 2,
        steps_per_epoch=100,
        validation_data=validation,
        # validation_steps=2,
        verbose=1,
    )


if __name__ == "__main__":
    typer.run(main)
