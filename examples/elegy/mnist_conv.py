from dataclasses import dataclass
import os
import typing as tp
from datetime import datetime
from functools import partial
from typing import Any, Generator, Mapping, Tuple

import dataget
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import typer
from tensorboardX.writer import SummaryWriter

import elegy as eg


@dataclass(unsafe_hash=True)
class ConvBlock(eg.Module):
    units: int
    kernel: tp.Tuple[int, int]
    stride: int = 1

    @eg.compact
    def __call__(self, x):
        x = eg.nn.Conv(
            self.units,
            self.kernel,
            strides=[self.stride, self.stride],
            padding="same",
        )(x)
        x = eg.nn.BatchNorm()(x)
        x = eg.nn.Dropout(0.2)(x)
        return jax.nn.relu(x)


class CNN(eg.Module):
    @eg.compact
    def __call__(self, image: jnp.ndarray):
        # normalize
        x: jnp.ndarray = image.astype(jnp.float32) / 255.0

        # base
        x = ConvBlock(32, (3, 3))(x)
        x = ConvBlock(64, (3, 3), stride=2)(x)
        x = ConvBlock(64, (3, 3), stride=2)(x)
        x = ConvBlock(128, (3, 3), stride=2)(x)

        # GlobalAveragePooling2D
        x = jnp.mean(x, axis=(1, 2))

        # 1x1 Conv
        x = eg.nn.Linear(10)(x)

        return x


def main(
    debug: bool = False,
    eager: bool = False,
    logdir: str = "runs",
    steps_per_epoch: tp.Optional[int] = None,
    epochs: int = 100,
    batch_size: int = 32,
):

    if debug:
        import debugpy

        print("Waiting for debugger...")
        debugpy.listen(5678)
        debugpy.wait_for_client()

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    logdir = os.path.join(logdir, current_time)

    X_train, y_train, X_test, y_test = dataget.image.mnist(global_cache=True).get()

    X_train = X_train[..., None]
    X_test = X_test[..., None]

    print("X_train:", X_train.shape, X_train.dtype)
    print("y_train:", y_train.shape, y_train.dtype)
    print("X_test:", X_test.shape, X_test.dtype)
    print("y_test:", y_test.shape, y_test.dtype)

    model = eg.Model(
        module=CNN(),
        loss=eg.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=eg.metrics.Accuracy(),
        optimizer=optax.adam(1e-3),
        eager=eager,
    )

    # show model summary
    model.summary(X_train[:64], depth=1)

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
