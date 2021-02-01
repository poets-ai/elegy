import os
from datetime import datetime

import dataget
import elegy
import jax
import jax.numpy as jnp
import numpy as np
import optax
import typer


class Model(elegy.Model):

    # request parameters by name via depending injection.
    # possible: net_params, x, y_true, net_states, metrics_states, sample_weight, class_weight, rng, states, initializing
    def test_step(
        self,
        x,
        y_true,
        net_params,
        states: elegy.States,
        initializing: bool,
        rng: elegy.RNGSeq,
    ):
        # flatten + scale
        x = jnp.reshape(x, (x.shape[0], -1)) / 255

        # model
        if initializing:
            w = jax.random.uniform(
                rng.next(), shape=[np.prod(x.shape[1:]), 10], minval=-1, maxval=1
            )
            b = jax.random.uniform(rng.next(), shape=[1], minval=-1, maxval=1)
            net_params = (w, b)

        w, b = net_params
        logits = jnp.dot(x, w) + b

        # crossentropy loss
        labels = jax.nn.one_hot(y_true, 10)
        loss = jnp.mean(-jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1))

        # metrics
        logs = dict(
            acc=jnp.mean(jnp.argmax(logits, axis=-1) == y_true),
            loss=loss,
        )

        return loss, logs, states.update(net_params=net_params)


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

    model = Model(optimizer=optax.adam(1e-3), run_eagerly=True)

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
