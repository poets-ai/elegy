from typing import Any, Generator, Mapping, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import typer
from jax.experimental import optix
import dataget

import elegy


def main(debug: bool = False, eager: bool = False):

    if debug:
        import debugpy

        print("Waiting for debugger...")
        debugpy.listen(5678)
        debugpy.wait_for_client()

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
            x = hk.Flatten()(image)
            x = hk.Sequential(
                [
                    hk.Linear(self.n1),
                    jax.nn.relu,
                    hk.Linear(self.n2),
                    jax.nn.relu,
                    hk.Linear(self.n1),
                    jax.nn.relu,
                    hk.Linear(x.shape[-1]),
                    jax.nn.sigmoid,
                ]
            )(x)
            return x.reshape(image.shape) * 255

    class MeanSquaredError(elegy.Loss):
        def call(self, x, y_pred):
            return jnp.mean(jnp.square(x - y_pred), axis=-1)

    model = elegy.Model(
        module=MLP.defer(n1=256, n2=64),
        loss=MeanSquaredError(),
        optimizer=optix.rmsprop(0.001),
        run_eagerly=eager,
    )

    history = model.fit(
        x=X_train, epochs=20, batch_size=64, validation_data=(X_test,), shuffle=True,
    )

    def plot_history(history):
        n_plots = len(history.history.keys()) // 2
        plt.figure(figsize=(14, 24))

        for i, key in enumerate(list(history.history.keys())[:n_plots]):
            if key == "size":
                continue

            metric = history.history[key]
            val_metric = history.history[f"val_{key}"]

            plt.subplot(n_plots, 1, i + 1)
            plt.plot(metric, label=f"Training {key}")
            plt.plot(val_metric, label=f"Validation {key}")
            plt.legend(loc="lower right")
            plt.ylabel(key)
            #         plt.ylim([min(plt.ylim()), 1])
            plt.title(f"Training and Validation {key}")
        plt.show()

    plot_history(history)

    # get random samples
    idxs = np.random.randint(0, 10000, size=(5,))
    x_sample = X_test[idxs]

    # get predictions
    y_pred = model.predict(x=x_sample)

    # plot results
    plt.figure(figsize=(12, 12))
    for i in range(5):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_sample[i], cmap="gray")
        plt.subplot(2, 5, 5 + i + 1)
        plt.imshow(y_pred[i], cmap="gray")

    plt.show()


if __name__ == "__main__":
    typer.run(main)
