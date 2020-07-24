from typing import Any, Generator, Mapping, Tuple

import dataget
import haiku as hk
import jax
import jax.numpy as jnp
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

    X_train = X_train[..., None]
    X_test = X_test[..., None]

    print("X_train:", X_train.shape, X_train.dtype)
    print("y_train:", y_train.shape, y_train.dtype)
    print("X_test:", X_test.shape, X_test.dtype)
    print("y_test:", y_test.shape, y_test.dtype)

    class CNN(elegy.Module):
        def __init__(self, n1: int = 300, n2: int = 100, **kwargs):
            super().__init__(**kwargs)
            self.n1 = n1
            self.n2 = n2

        def call(self, image: jnp.ndarray, is_training: bool):
            def conv_block(x, units, kernel, stride=1):
                x = hk.Conv2D(units, kernel, stride=stride, padding="same")(x)
                x = elegy.nn.BatchNormalization()(x, is_training)
                x = elegy.nn.Dropout(0.2)(x, is_training)
                return jax.nn.relu(x)

            x: np.ndarray = image.astype(jnp.float32) / 255.0

            # base
            x = conv_block(x, 16, [3, 3])
            x = conv_block(x, 32, [3, 3], stride=2)
            x = conv_block(x, 64, [3, 3], stride=2)
            x = conv_block(x, 64, [3, 3], stride=2)

            # 1x1 Conv
            x = hk.Linear(10)(x)

            # GlobalAveragePooling2D
            x = jnp.mean(x, axis=[1, 2])

            return x

    model = elegy.Model(
        module=CNN.defer(n1=300, n2=100),
        loss=elegy.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=elegy.metrics.SparseCategoricalAccuracy.defer(),
        optimizer=optix.rmsprop(1e-3),
        run_eagerly=eager,
    )

    # force initialize
    model.predict(X_train[:64])

    print(
        f"Total parameters: {sum(np.prod(x.shape) for x in jax.tree_leaves(model.params)):,}"
    )

    history = model.fit(
        x=X_train,
        y=y_train,
        epochs=100,
        steps_per_epoch=200,
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
