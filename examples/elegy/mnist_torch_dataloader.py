import os
import typing as tp
from datetime import datetime

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import torch
import typer
from datasets.load import load_dataset
from tensorboardX.writer import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset

import elegy as eg


@eg.compact_module
def ConvBlock(
    x,
    units: int,
    kernel: tp.Tuple[int, int],
    stride: int = 1,
):
    x = eg.Conv(
        units,
        kernel,
        strides=[stride, stride],
        padding="same",
    )(x)
    x = eg.BatchNorm()(x)
    x = eg.Dropout(0.2)(x)
    return jax.nn.relu(x)


class CNN(eg.Module):
    @eg.compact
    def __call__(self, x: jnp.ndarray):
        # normalize
        x = x.astype(jnp.float32) / 255.0

        # base
        x = ConvBlock()(x, 32, (3, 3))
        x = ConvBlock()(x, 64, (3, 3), stride=2)
        x = ConvBlock()(x, 64, (3, 3), stride=2)
        x = ConvBlock()(x, 128, (3, 3), stride=2)

        # GlobalAveragePooling2D
        x = jnp.mean(x, axis=(1, 2))

        # 1x1 Conv
        x = eg.Linear(10)(x)

        return x


def main(
    debug: bool = False,
    eager: bool = False,
    logdir: str = "runs",
    steps_per_epoch: int = 200,
    epochs: int = 100,
    batch_size: int = 64,
):

    if debug:
        import debugpy

        print("Waiting for debugger...")
        debugpy.listen(5678)
        debugpy.wait_for_client()

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    logdir = os.path.join(logdir, current_time)

    dataset = load_dataset("mnist")
    dataset.set_format("np")
    X_train = dataset["train"]["image"]
    y_train = dataset["train"]["label"]
    X_test = dataset["test"]["image"]
    y_test = dataset["test"]["label"]

    X_train = X_train[..., None]
    X_test = X_test[..., None]

    print("X_train:", X_train.shape, X_train.dtype)
    print("y_train:", y_train.shape, y_train.dtype)
    print("X_test:", X_test.shape, X_test.dtype)
    print("y_test:", y_test.shape, y_test.dtype)

    model = eg.Model(
        module=CNN(),
        loss=eg.losses.Crossentropy(),
        metrics=eg.metrics.Accuracy(),
        optimizer=optax.adam(1e-3),
        eager=eager,
    )

    # show summary
    model.summary(X_train[:64])

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    history = model.fit(
        train_dataloader,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_dataloader,
        callbacks=[eg.callbacks.TensorBoard(logdir=logdir)],
    )

    eg.utils.plot_history(history)

    model.save("models/conv")

    model = eg.load("models/conv")

    print(model.evaluate(x=X_test, y=y_test))

    # get random samples
    idxs = np.random.randint(0, 10000, size=(9,))
    x_sample = X_test[idxs]

    # get predictions
    y_pred = model.predict(x=x_sample)

    # plot results
    with SummaryWriter(os.path.join(logdir, "val")) as tbwriter:
        figure = plt.figure(figsize=(12, 12))
        for i in range(3):
            for j in range(3):
                k = 3 * i + j
                plt.subplot(3, 3, k + 1)

                plt.title(f"{np.argmax(y_pred[k])}")
                plt.imshow(x_sample[k], cmap="gray")
        # tbwriter.add_figure("Conv classifier", figure, 100)

    plt.show()


if __name__ == "__main__":
    typer.run(main)
