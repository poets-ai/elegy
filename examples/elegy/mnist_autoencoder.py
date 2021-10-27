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

import elegy as eg


class MeanSquaredError(eg.losses.MeanSquaredError):
    # we request `x` instead of `y_true` since we are don't require labels in autoencoders
    def call(self, inputs, preds):
        return super().call(target=inputs, preds=preds) / 255


class MLP(eg.Module):
    def __init__(self, n1: int = 300, n2: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.n1 = n1
        self.n2 = n2

    @eg.compact
    def __call__(self, image: jnp.ndarray):
        x = image.astype(jnp.float32) / 255.0
        x = eg.nn.Flatten()(x)
        x = eg.nn.Linear(self.n1)(x)
        x = jax.nn.relu(x)
        x = eg.nn.Linear(self.n2)(x)
        x = jax.nn.relu(x)
        x = eg.nn.Linear(self.n1)(x)
        x = jax.nn.relu(x)
        x = eg.nn.Linear(np.prod(image.shape[-2:]))(x)
        x = jax.nn.sigmoid(x) * 255
        x = x.reshape(image.shape)

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

    X_train, _1, X_test, _2 = dataget.image.mnist(global_cache=True).get()

    print("X_train:", X_train.shape, X_train.dtype)
    print("X_test:", X_test.shape, X_test.dtype)

    model = eg.Model(
        module=MLP(n1=256, n2=64),
        loss=MeanSquaredError(),
        optimizer=optax.rmsprop(0.001),
        eager=eager,
    )

    model.summary(X_train[:64])

    # Notice we are not passing `y`
    history = model.fit(
        inputs=X_train,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        validation_data=(X_test,),
        shuffle=True,
        callbacks=[eg.callbacks.TensorBoard(logdir=logdir, update_freq=300)],
    )

    eg.utils.plot_history(history)

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

    plt.show()


if __name__ == "__main__":
    typer.run(main)
