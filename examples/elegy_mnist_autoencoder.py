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

    X_train, _1, X_test, _2 = dataget.image.mnist(global_cache=True).get()

    print("X_train:", X_train.shape, X_train.dtype)
    print("X_test:", X_test.shape, X_test.dtype)

    class MLP(elegy.Module):
        """Standard LeNet-300-100 MLP network."""

        def __init__(self, n1: int = 300, n2: int = 100, **kwargs):
            super().__init__(**kwargs)
            self.n1 = n1
            self.n2 = n2

        def call(self, image: jnp.ndarray):
            image = image.astype(jnp.float32) / 255.0
            x = elegy.nn.Flatten()(image)
            x = elegy.nn.sequential(
                elegy.nn.Linear(self.n1),
                jax.nn.relu,
                elegy.nn.Linear(self.n2),
                jax.nn.relu,
                elegy.nn.Linear(self.n1),
                jax.nn.relu,
                elegy.nn.Linear(x.shape[-1]),
                jax.nn.sigmoid,
            )(x)
            return x.reshape(image.shape) * 255

    class MeanSquaredError(elegy.losses.MeanSquaredError):
        # we request `x` instead of `y_true` since we are don't require labels in autoencoders
        def call(self, x, y_pred):
            return super().call(x, y_pred)

    model = elegy.Model(
        module=MLP(n1=256, n2=64),
        loss=MeanSquaredError(),
        optimizer=optax.rmsprop(0.001),
        run_eagerly=eager,
    )

    model.summary(X_train[:64])

    # Notice we are not passing `y`
    history = model.fit(
        x=X_train,
        epochs=epochs,
        batch_size=64,
        validation_data=(X_test,),
        shuffle=True,
        callbacks=[elegy.callbacks.TensorBoard(logdir=logdir, update_freq=300)],
    )

    plot_history(history)

    # get random samples
    idxs = np.random.randint(0, 10000, size=(5,))
    x_sample = X_test[idxs]

    # get predictions
    y_pred = model.predict(x=x_sample)

    # plot and save results
    with SummaryWriter(os.path.join(logdir, "val")) as tbwriter:

        figure = plt.figure(figsize=(12, 12))
        for i in range(5):
            plt.subplot(2, 5, i + 1)
            plt.imshow(x_sample[i], cmap="gray")
            plt.subplot(2, 5, 5 + i + 1)
            plt.imshow(y_pred[i], cmap="gray")

        # tbwriter.add_figure("AutoEncoder images", figure, 20)

    plt.show()

    print(
        "\n\n\nMetrics and images can be explored using tensorboard using:",
        f"\n \t\t\t tensorboard --logdir {logdir}",
    )


if __name__ == "__main__":
    typer.run(main)
