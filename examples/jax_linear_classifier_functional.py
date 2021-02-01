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


def main(
    debug: bool = False,
    eager: bool = False,
    logdir: str = "runs",
    steps_per_epoch: int = 200,
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

    def linear_classifier(
        x: jnp.ndarray, net_params, rng, initializing
    ) -> elegy.OutputStates:
        x = x.reshape((x.shape[0], -1)) / 255

        if initializing:
            w = jax.random.uniform(rng.next(), shape=[x.shape[-1], 10])
            b = jax.random.uniform(rng.next(), shape=[10])
            net_params = (w, b)

        w, b = net_params
        y_pred = jnp.dot(x, w) + b

        return elegy.OutputStates(y_pred, net_params, None)

    model = elegy.Model(
        module=linear_classifier,
        loss=crossentropy,
        metrics=accuracy,
        optimizer=optax.adam(1e-3),
        run_eagerly=eager,
    )

    history = model.fit(
        x=X_train,
        y=y_train,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=64,
        validation_data=(X_test, y_test),
        shuffle=True,
        callbacks=[elegy.callbacks.TensorBoard(logdir=logdir)],
    )

    elegy.utils.plot_history(history)


if __name__ == "__main__":
    typer.run(main)
