from typing import Any, Generator, Mapping, Tuple

import dataget
import haiku as hk
import jax
import jax.numpy as jnp
from jax.numpy.lax_numpy import mod
import matplotlib.pyplot as plt
import numpy as np
import typer
from jax.experimental import optix

import elegy
from utils import plot_history


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

        def call(self, image: jnp.ndarray, is_training: bool):

            image = image.astype(jnp.float32) / 255.0

            mlp = hk.Sequential(
                [
                    hk.Flatten(),
                    hk.Linear(self.n1),
                    lambda x: elegy.nn.BatchNormalization()(x, is_training),
                    jax.nn.relu,
                    hk.Linear(self.n2),
                    jax.nn.relu,
                    hk.Linear(10),
                ]
            )

            if is_training:
                elegy.add_loss("a", 1.0)
                elegy.add_metric("b", 5.0)
            else:
                elegy.add_loss("a", 2.0)
                elegy.add_metric("b", 10.0)

            return mlp(image)

    model = elegy.Model(
        module=MLP.defer(n1=300, n2=100),
        loss=[
            elegy.losses.SparseCategoricalCrossentropy(from_logits=True),
            elegy.regularizers.GlobalL2(l=1e-4),
        ],
        metrics=elegy.metrics.SparseCategoricalAccuracy.defer(),
        optimizer=optix.rmsprop(1e-3),
        run_eagerly=eager,
    )

    model.summary(X_train[:64])

    history = model.fit(
        x=X_train,
        y=y_train,
        epochs=10,
        steps_per_epoch=20,
        batch_size=64,
        validation_data=(X_test, y_test),
        shuffle=True,
    )

    plot_history(history)

    # get random samples
    idxs = np.random.randint(0, 10000, size=(9,))
    x_sample = X_test[idxs]

    # get predictions
    y_pred = model.predict(x=x_sample)

    # plot results
    plt.figure(figsize=(12, 12))
    for i in range(3):
        for j in range(3):
            k = 3 * i + j
            plt.subplot(3, 3, k + 1)

            plt.title(f"{np.argmax(y_pred[k])}")
            plt.imshow(x_sample[k], cmap="gray")

    plt.show()


if __name__ == "__main__":
    typer.run(main)
