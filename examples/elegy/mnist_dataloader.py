import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Generator, Mapping, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import typer
from datasets.load import load_dataset
from tensorboardX.writer import SummaryWriter

import elegy as eg


class MNIST(eg.data.Dataset):
    def __init__(self, training: bool = True):

        dataset = load_dataset("mnist")
        dataset.set_format("np")
        X_train = dataset["train"]["image"]
        y_train = dataset["train"]["label"]
        X_test = dataset["test"]["image"]
        y_test = dataset["test"]["label"]

        if training:
            self.x = X_train
            self.y = y_train
        else:
            self.x = X_test
            self.y = y_test

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return (self.x[i], self.y[i])


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

    train_dataset = MNIST(training=True)
    test_dataset = MNIST(training=False)
    train_loader = eg.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = eg.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print("X_train:", train_dataset.x.shape, train_dataset.x.dtype)
    print("y_train:", train_dataset.y.shape, train_dataset.y.dtype)
    print("X_test:", test_dataset.x.shape, test_dataset.x.dtype)
    print("y_test:", test_dataset.y.shape, test_dataset.y.dtype)

    @dataclass(unsafe_hash=True, repr=False)
    class MLP(eg.Module):
        """Standard LeNet-300-100 MLP network."""

        n1: int = 300
        n2: int = 100

        @eg.compact
        def __call__(self, x: jnp.ndarray):
            x = x.astype(jnp.float32) / 255.0

            x = eg.Flatten()(x)
            x = eg.Linear(self.n1)(x)
            x = jax.nn.relu(x)
            x = eg.Linear(self.n2)(x)
            x = jax.nn.relu(x)
            x = eg.Linear(10)(x)

            return x

    model = eg.Model(
        module=MLP(n1=300, n2=100),
        loss=[
            eg.losses.Crossentropy(),
            eg.regularizers.L2(l=1e-4),
        ],
        metrics=eg.metrics.Accuracy(),
        optimizer=optax.adamw(1e-3),
        eager=eager,
    )

    x_sample, y_sample = next(iter(train_loader))
    model.summary(x_sample)

    history = model.fit(
        inputs=train_loader,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_loader,
        shuffle=True,
        callbacks=[eg.callbacks.TensorBoard(logdir=logdir)],
    )

    eg.utils.plot_history(history)

    # get random samples
    idxs = np.random.randint(0, 10000, size=(9,))
    x_sample, y_sample = next(iter(test_loader))

    # get predictions
    y_pred = model.predict(x=x_sample)

    # plot and save results
    def make_plot():
        plt.figure(figsize=(12, 12))
        for i in range(3):
            for j in range(3):
                k = 3 * i + j
                plt.subplot(3, 3, k + 1)
                plt.title(f"{np.argmax(y_pred[k])}")
                plt.imshow(x_sample[k], cmap="gray")

    with SummaryWriter(os.path.join(logdir, "val")) as tbwriter:
        make_plot()
        # tbwriter.add_figure("Predictions", plt.gcf(), 100)

    make_plot()
    plt.show()

    print(
        "\n\n\nMetrics and images can be explored using tensorboard using:",
        f"\n \t\t\t tensorboard --logdir {logdir}",
    )


if __name__ == "__main__":
    typer.run(main)
