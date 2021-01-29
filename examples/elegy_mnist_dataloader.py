import os
from datetime import datetime
from typing import Any, Generator, Mapping, Tuple

import dataget

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX.writer import SummaryWriter
import typer
import optax

import elegy
from utils import plot_history


class MNIST(elegy.data.Dataset):
    def __init__(self, training: bool = True):

        X_train, y_train, X_test, y_test = dataget.image.mnist(global_cache=True).get()

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
    debug: bool = False, eager: bool = False, logdir: str = "runs", epochs: int = 100
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
    train_loader = elegy.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = elegy.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

    print("X_train:", train_dataset.x.shape, train_dataset.x.dtype)
    print("y_train:", train_dataset.y.shape, train_dataset.y.dtype)
    print("X_test:", test_dataset.x.shape, test_dataset.x.dtype)
    print("y_test:", test_dataset.y.shape, test_dataset.y.dtype)

    class MLP(elegy.Module):
        """Standard LeNet-300-100 MLP network."""

        def __init__(self, n1: int = 300, n2: int = 100, **kwargs):
            super().__init__(**kwargs)
            self.n1 = n1
            self.n2 = n2

        def call(self, image: jnp.ndarray):
            image = image.astype(jnp.float32) / 255.0

            mlp = elegy.nn.sequential(
                elegy.nn.Flatten(),
                elegy.nn.Linear(self.n1),
                jax.nn.relu,
                elegy.nn.Linear(self.n2),
                jax.nn.relu,
                elegy.nn.Linear(10),
            )

            return mlp(image)

    model = elegy.Model(
        module=MLP(n1=300, n2=100),
        loss=[
            elegy.losses.SparseCategoricalCrossentropy(from_logits=True),
            elegy.regularizers.GlobalL2(l=1e-4),
        ],
        metrics=elegy.metrics.SparseCategoricalAccuracy(),
        optimizer=optax.adamw(1e-3),
        run_eagerly=eager,
    )

    x_sample, y_sample = next(iter(train_loader))
    model.summary(x_sample)

    history = model.fit(
        x=train_loader,
        epochs=epochs,
        steps_per_epoch=200,
        validation_data=test_loader,
        shuffle=True,
        callbacks=[elegy.callbacks.TensorBoard(logdir=logdir)],
    )

    plot_history(history)

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
        tbwriter.add_figure("Predictions", plt.gcf(), 100)

    make_plot()
    plt.show()

    print(
        "\n\n\nMetrics and images can be explored using tensorboard using:",
        f"\n \t\t\t tensorboard --logdir {logdir}",
    )


if __name__ == "__main__":
    typer.run(main)
