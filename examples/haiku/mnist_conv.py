import os
import typing as tp
from datetime import datetime

import datasets
import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import typer
from datasets import load_dataset

import elegy as eg


def forward(x: jnp.ndarray, training: bool):
    # Normalize input
    x = x.astype(jnp.float32) / 255.0

    # Block 1
    x = hk.Conv2D(32, [3, 3], stride=[2, 2])(x)
    x = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.99)(
        x, is_training=training
    )
    x = hk.dropout(hk.next_rng_key(), 0.05, x)
    x = jax.nn.relu(x)

    # Block 2
    x = hk.Conv2D(64, [3, 3], stride=[2, 2])(x)
    x = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.99)(
        x, is_training=training
    )
    x = hk.dropout(hk.next_rng_key(), 0.1, x)
    x = jax.nn.relu(x)

    # Block 3
    x = hk.Conv2D(128, [3, 3], stride=[2, 2])(x)

    # GlobalAveragePooling2D
    x = x.mean(axis=(1, 2))

    # Classification layer
    x = hk.Linear(10)(x)

    return x


def main(
    debug: bool = False,
    eager: bool = False,
    logdir: str = "runs",
    steps_per_epoch: tp.Optional[int] = None,
    epochs: int = 100,
    batch_size: int = 32,
    distributed: bool = False,
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
    X_train = np.stack(dataset["train"]["image"])[..., None]
    y_train = dataset["train"]["label"]
    X_test = np.stack(dataset["test"]["image"])[..., None]
    y_test = dataset["test"]["label"]

    print("X_train:", X_train.shape, X_train.dtype)
    print("y_train:", y_train.shape, y_train.dtype)
    print("X_test:", X_test.shape, X_test.dtype)
    print("y_test:", y_test.shape, y_test.dtype)

    model = eg.Model(
        module=hk.transform_with_state(forward),
        loss=eg.losses.Crossentropy(),
        metrics=eg.metrics.Accuracy(),
        optimizer=optax.adam(1e-3),
        eager=eager,
    )

    if distributed:
        model = model.distributed()

    model.summary(X_train[:batch_size])

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

    print(model.evaluate(x=X_test, y=y_test))

    # get random samples
    idxs = np.random.randint(0, 10000, size=(9,))
    x_sample = X_test[idxs]

    # get predictions
    model = model.local()
    y_pred = model.predict(x=x_sample)

    # plot results
    figure = plt.figure(figsize=(12, 12))
    for i in range(3):
        for j in range(3):
            k = 3 * i + j
            plt.subplot(3, 3, k + 1)

            plt.title(f"{np.argmax(y_pred[k])}")
            plt.imshow(x_sample[k], cmap="gray")

    plt.show()


if __name__ == "__main__":
    typer.run(main)
