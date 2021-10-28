import os
from datetime import datetime

import datasets
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import typer
from datasets import load_dataset
from flax import linen

import elegy


class LinearClassifier(linen.Module):
    @linen.compact
    def __call__(self, x):
        x = jnp.reshape(x, (x.shape[0], -1)) / 255.0
        x = linen.Dense(10)(x)
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

    print("X_train:", X_train.shape, X_train.dtype)
    print("y_train:", y_train.shape, y_train.dtype)
    print("X_test:", X_test.shape, X_test.dtype)
    print("y_test:", y_test.shape, y_test.dtype)

    model = elegy.Model(
        module=elegy.FlaxModule(LinearClassifier()),
        loss=elegy.losses.Crossentropy(),
        metrics=elegy.metrics.Accuracy(),
        optimizer=optax.adam(1e-3),
        eager=eager,
    )

    model.summary(X_train[:batch_size])

    history = model.fit(
        inputs=X_train,
        labels=y_train,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        shuffle=True,
        callbacks=[elegy.callbacks.TensorBoard(logdir=logdir)],
    )

    elegy.utils.plot_history(history)

    plt.show()


if __name__ == "__main__":
    typer.run(main)
