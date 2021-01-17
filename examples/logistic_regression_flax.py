import os
from datetime import datetime

import dataget
from flax import linen
import elegy
import jax
import jax.numpy as jnp
import numpy as np
import optax
import typer

from utils import plot_history


def main(
    debug: bool = False,
    eager: bool = False,
    logdir: str = "runs",
    epochs: int = 100,
):

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

    def crossentropy(y_true, y_pred):
        labels = jax.nn.one_hot(y_true, 10)
        loss = -jnp.sum(labels * jax.nn.log_softmax(y_pred), axis=-1)
        return jnp.mean(loss)

    def accuracy(y_true, y_pred):
        return jnp.mean(jnp.argmax(y_pred, axis=-1) == y_true)

    class LogisticRegression(linen.Module):
        @linen.compact
        def __call__(self, x):
            x = jnp.reshape(x, (x.shape[0], -1)) / 255.0
            x = linen.Dense(10)(x)
            return x

    model = elegy.Model(
        module=LogisticRegression(),
        loss=crossentropy,
        metrics=accuracy,
        optimizer=optax.adam(1e-3),
    )

    history = model.fit(
        x=X_train,
        y=y_train,
        epochs=epochs,
        steps_per_epoch=200,
        batch_size=64,
        validation_data=(X_test, y_test),
        shuffle=True,
        callbacks=[elegy.callbacks.TensorBoard(logdir=logdir)],
    )

    plot_history(history)


if __name__ == "__main__":
    typer.run(main)
