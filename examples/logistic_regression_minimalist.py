import os
from datetime import datetime

import dataget
import elegy
import jax
import jax.numpy as jnp
import numpy as np
import optax
import typer

from utils import plot_history


class Model(elegy.Model):
    # request parameters by name via depending injection.
    # possible: mode, x, y_true, sample_weight, class_weight
    def init(self, x):
        d = np.prod(x.shape[1:])

        # friendly RNG interface: rng.next() == jax.random.split(...)
        rng = elegy.RNGSeq(42)

        # params
        w = jax.random.uniform(rng.next(), shape=[d, 10], minval=-1, maxval=1)
        b = jax.random.uniform(rng.next(), shape=[1], minval=-1, maxval=1)

        return elegy.States(
            net_params=(w, b),
            rng=rng,
        )

    # request parameters by name via depending injection.
    # possible: net_params, x, y_true, net_states, metrics_states, optimizer_states, sample_weight, class_weight, rng
    def train_step(self, x, y_true, net_params):
        # flatten + scale
        x = jnp.reshape(x, (x.shape[0], -1)) / 255

        def loss_fn(net_params, x, y_true):
            # model
            w, b = net_params
            logits = jnp.dot(x, w) + b

            # crossentropy loss
            labels = jax.nn.one_hot(y_true, 10)
            loss = jnp.mean(-jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1))

            return loss, logits

        # train
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

        (loss, logits), grads = grad_fn(net_params, x, y_true)

        # sgd
        net_params = jax.tree_multimap(lambda p, g: p - 3e-4 * g, net_params, grads)

        # metrics
        logs = dict(
            acc=jnp.mean(jnp.argmax(logits, axis=-1) == y_true),
            loss=loss,
        )

        return logs, elegy.States(
            net_params=net_params,
        )


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

    X_train, y_train, X_test, y_test = dataget.image.mnist(global_cache=True).get()

    print("X_train:", X_train.shape, X_train.dtype)
    print("y_train:", y_train.shape, y_train.dtype)
    print("X_test:", X_test.shape, X_test.dtype)
    print("y_test:", y_test.shape, y_test.dtype)

    model = Model()

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
