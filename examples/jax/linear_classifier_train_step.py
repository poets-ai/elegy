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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.optim = optax.adam(1e-3)

    # request parameters by name via depending injection.
    # possible: initializing, net_params, x, y_true, net_states, metrics_states, optimizer_states, sample_weight, class_weight, rng
    def train_step(self, x, y_true, initializing, states: elegy.States):
        def loss_fn(net_params, x, y_true):
            assert isinstance(states.rng, elegy.KeySeq)

            # flatten + scale
            x = jnp.reshape(x, (x.shape[0], -1)) / 255

            # model
            w, b = net_params
            logits = jnp.dot(x, w) + b

            # crossentropy loss
            labels = jax.nn.one_hot(y_true, 10)
            sample_loss = -jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1)
            loss = jnp.mean(sample_loss)

            # metrics
            logs = dict(
                accuracy=jnp.mean(jnp.argmax(logits, axis=-1) == y_true),
                loss=loss,
            )

            return loss, logs

        if initializing:
            w = jax.random.uniform(
                states.rng.next(), shape=[np.prod(x.shape[1:]), 10], minval=-1, maxval=1
            )
            b = jax.random.uniform(states.rng.next(), shape=[1], minval=-1, maxval=1)
            states = states.update(net_params=(w, b))

        # train
        (loss, logs), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            states.net_params,  # gradients target
            x,
            y_true,
        )

        if initializing:
            optimizer_states = self.optim.init(states.net_params)
            net_params = states.net_params
        else:
            grads, optimizer_states = self.optim.update(
                grads, states.optimizer_states, states.net_params
            )
            net_params = optax.apply_updates(states.net_params, grads)

        return logs, states.update(
            net_params=net_params,
            optimizer_states=optimizer_states,
        )


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

    X_train, y_train, X_test, y_test = dataget.image.mnist(global_cache=True).get()

    print("X_train:", X_train.shape, X_train.dtype)
    print("y_train:", y_train.shape, y_train.dtype)
    print("X_test:", X_test.shape, X_test.dtype)
    print("y_test:", y_test.shape, y_test.dtype)

    model = Model(eager=eager)

    history = model.fit(
        inputs=X_train,
        labels=y_train,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        shuffle=True,
        callbacks=[elegy.callbacks.TensorBoard(logdir=logdir)],
    )

    elegy.utils.plot_history(history)


if __name__ == "__main__":
    typer.run(main)
