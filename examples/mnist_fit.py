from typing import Any, Generator, Mapping, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds
import typer

import elegy

OptState = Any
Batch = Mapping[str, np.ndarray]
np.random.seed(42)


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


def net_fn(image) -> jnp.ndarray:
    """Standard LeNet-300-100 MLP network."""
    image = image.astype(jnp.float32) / 255.0

    mlp = hk.Sequential(
        [
            hk.Flatten(),
            hk.Linear(300),
            jax.nn.relu,
            hk.Linear(100),
            jax.nn.relu,
            hk.Linear(10),
        ]
    )
    return mlp(image)


def loss_fn(y_true, y_pred, params) -> jnp.ndarray:

    l2_loss = elegy.losses.L2Regularization()(params)
    softmax_xent = elegy.losses.SoftmaxCrossentropy()(y_true, y_pred)

    return softmax_xent + 1e-4 * l2_loss


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

    n_iters = 100
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

    model = elegy.Model(
        model_fn=net_fn,
        loss=loss_fn,
        metrics=lambda: elegy.metrics.Accuracy(),
        run_eagerly=eager,
    )

    # Fit with datasets in memory
    # model.fit(
    #     x=x, y=y, epochs=10, batch_size=60, validation_data=(x_val, y_val),
    # )
    # exit()

    # Fit with generators
    x = load_dataset("train", is_training=True, batch_size=64, for_fit=True)
    validation = load_dataset("train", is_training=False, batch_size=1000, for_fit=True)

    model.fit(
        x,
        epochs=10,
        steps_per_epoch=106,
        validation_data=validation,
        validation_steps=2,
    )


if __name__ == "__main__":
    typer.run(main)
