import os
from datetime import datetime
from functools import partial
from typing import Any, Generator, Mapping, Tuple

import einops
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import typer
from datasets.load import load_dataset
from tensorboardX.writer import SummaryWriter

import elegy as eg


class MLP(eg.CoreModule):
    def __init__(self, n1: int = 300, n2: int = 100):
        self.n1 = n1
        self.n2 = n2

    @eg.compact
    def __call__(self, x: jnp.ndarray):
        x = x.astype(jnp.float32) / 255.0
        x = einops.rearrange(x, "batch ... -> batch (...)")
        x = eg.nn.Linear(self.n1)(x)
        x = jax.nn.relu(x)
        x = eg.nn.Linear(self.n2)(x)
        x = jax.nn.relu(x)
        x = eg.nn.Linear(10)(x)
        return x


def main(
    debug: bool = False,
    eager: bool = False,
    logdir: str = "runs",
    steps_per_epoch: int = 200,
    batch_size: int = 64,
    epochs: int = 100,
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
    X_train = np.stack(dataset["train"]["image"])
    y_train = dataset["train"]["label"]
    X_test = np.stack(dataset["test"]["image"])
    y_test = dataset["test"]["label"]

    print("X_train:", X_train.shape, X_train.dtype)
    print("y_train:", y_train.shape, y_train.dtype)
    print("X_test:", X_test.shape, X_test.dtype)
    print("y_test:", y_test.shape, y_test.dtype)

    model = eg.Trainer(
        module=MLP(n1=300, n2=100),
        loss=[
            eg.losses.Crossentropy(),
            eg.regularizers.L2(l=1e-4),
        ],
        metrics=eg.metrics.Accuracy(),
        optimizer=optax.adamw(1e-3),
        eager=eager,
    )

    model.summary(X_train[:64])

    history = model.fit(
        inputs=X_train,
        labels=y_train,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        shuffle=True,
        callbacks=[eg.callbacks.TensorBoard(logdir=logdir)],
    )

    eg.utils.plot_history(history)

    # get random samples
    idxs = np.random.randint(0, 10000, size=(9,))
    x_sample = X_test[idxs]

    # get predictions
    y_pred = model.predict(x=x_sample)

    # plot and save results
    with SummaryWriter(os.path.join(logdir, "val")) as tbwriter:
        figure = plt.figure(figsize=(12, 12))
        for i in range(3):
            for j in range(3):
                k = 3 * i + j
                plt.subplot(3, 3, k + 1)
                plt.title(f"{np.argmax(y_pred[k])}")
                plt.imshow(x_sample[k], cmap="gray")
        # tbwriter.add_figure("Predictions", figure, 100)

    plt.show()

    print(
        "\n\n\nMetrics and images can be explored using tensorboard using:",
        f"\n \t\t\t tensorboard --logdir {logdir}",
    )


if __name__ == "__main__":
    typer.run(main)
