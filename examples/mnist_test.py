from elegy.utils import Mode
from elegy.model.model_base import ModelBase
import os
from datetime import datetime
from typing import Any, Generator, Mapping, Tuple

import dataget

import jax
import jax.numpy as jnp
from jax.numpy.lax_numpy import mod
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX.writer import SummaryWriter
import typer
import optax

import elegy
from utils import plot_history


def random_batch(x, y, batch_size):
    idx = np.random.choice(len(x), batch_size)
    return x[idx], y[idx]


def main(debug: bool = False, eager: bool = False, logdir: str = "runs"):

    if debug:
        import debugpy

        print("Waiting for debugger...")
        debugpy.listen(5678)
        debugpy.wait_for_client()

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    logdir = os.path.join(logdir, current_time)

    X_train, y_train, X_test, y_test = dataget.image.mnist(global_cache=True).get()

    print("X_train:", X_train.shape, X_train.dtype)
    print("y_train:", y_train.shape, y_train.dtype)
    print("X_test:", X_test.shape, X_test.dtype)
    print("y_test:", y_test.shape, y_test.dtype)

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

    model = ModelBase(
        module=MLP(n1=300, n2=100),
        loss=[
            elegy.losses.SparseCategoricalCrossentropy(from_logits=True),
            elegy.regularizers.GlobalL2(l=1e-4),
        ],
        metrics=elegy.metrics.SparseCategoricalAccuracy(),
        optimizer=optax.adamw(1e-3),
        # run_eagerly=eager,
    )

    model.maybe_initialize(
        mode=Mode.train,
        x=X_train[:64],
        y=y_train[:64],
    )
    model.module.summary(X_train[:64])

    for epoch in range(100):

        model.reset_metrics()
        for step in range(200):
            x_sample, y_sample = random_batch(X_train, y_train, 64)

            logs = model.train_step_jit(
                x_sample, y_sample, sample_weight=None, class_weight=None
            )

            print(
                f"[TRAIN] epoch: {epoch},",
                ", ".join(f"{k}: {float(v):.3f}" for k, v in logs.items()),
            )

        model.reset_metrics()
        for test_step in range(10):
            x_sample, y_sample = random_batch(X_test, y_test, 64)
            loss, logs, grads = model.test_step_jit(
                x_sample, y_sample, sample_weight=None, class_weight=None
            )

        print(
            f"[TEST] epoch: {epoch},",
            ", ".join(f"{k}: {float(v):.3f}" for k, v in logs.items()),
        )

    exit()

    history = model.fit(
        x=X_train,
        y=y_train,
        epochs=100,
        steps_per_epoch=200,
        batch_size=64,
        validation_data=(X_test, y_test),
        shuffle=True,
        callbacks=[elegy.callbacks.TensorBoard(logdir=logdir)],
    )

    print(model.module.submodules)

    plot_history(history)

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
        tbwriter.add_figure("Predictions", figure, 100)

    plt.show()

    print(
        "\n\n\nMetrics and images can be explored using tensorboard using:",
        f"\n \t\t\t tensorboard --logdir {logdir}",
    )


if __name__ == "__main__":
    typer.run(main)
