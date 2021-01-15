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

import elegy
from utils import plot_history
import flax.linen as nn


class Mean(nn.Module):
    def maybe_intializing(self):
        return len(self.scope._variables) == 0

    @nn.compact
    def __call__(self, value: np.ndarray):
        initializing = self.maybe_intializing()

        vn = self.variable("metrics", "n", lambda: jnp.array(0, dtype=jnp.int32))
        vtotal = self.variable(
            "metrics", "total", lambda: jnp.array(0.0, dtype=jnp.float32)
        )

        n = vn.value + np.prod(value.shape)
        total = vtotal.value + np.sum(value)

        if not initializing:
            vn.value = n
            vtotal.value = total

        return total / n


class Model(elegy.ModelBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.optimizer = optax.adam(1e-3)

    def init(self, x):
        # friendly RNG interface: next == split
        rng = elegy.RNGSeq(42)

        # pred
        w = jax.random.uniform(rng.next(), shape=[1, np.prod(x.shape[1:])])
        b = jax.random.uniform(rng.next(), shape=[1])
        net_params = (w, b)

        # test
        total_samples = jnp.array(0, dtype=jnp.float32)
        total_acc = jnp.array(0, dtype=jnp.float32)
        total_loss = jnp.array(0, dtype=jnp.float32)

        # train
        optimizer_states = self.optimizer.init(net_params)

        return elegy.States(
            net_params=net_params,
            metrics_states=(total_samples, total_acc, total_loss),
            optimizer_states=optimizer_states,
            rng=rng,
        )

    def train_step(self, x, y_true, net_params, metrics_states, optimizer_states):
        def loss_fn(net_params, x, y_true):
            # model
            w, b = net_params
            logits = jnp.dot(x, w) + b

            # binary crossentropy loss
            sample_loss = -jnp.sum(
                y_true * logits - jnp.logaddexp(0.0, logits),
                axis=-1,
            )
            loss = jnp.mean(sample_loss)

            return loss, (logits, sample_loss)

        # train
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, (logits, sample_loss)), grads = grad_fn(net_params, x, y_true)

        grads, optimizer_states = self.optimizer.update(
            grads, optimizer_states, net_params
        )
        net_params = optax.apply_updates(net_params, grads)

        # metrics
        sample_acc = logits[..., 0] >= 0 == y_true

        total_samples, total_acc, total_loss = metrics_states
        total_samples += x.shape[0]
        total_acc += jnp.sum(sample_acc)
        total_loss += jnp.sum(sample_loss)

        logs = dict(
            acc=total_acc / total_samples,
            loss=total_loss / total_samples,
        )

        return logs, elegy.States(
            net_params=net_params,
            metrics_states=(total_samples, total_acc, total_loss),
            optimizer_states=optimizer_states,
        )


def main(debug: bool = False, eager: bool = False, logdir: str = "runs"):

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
        epochs=100,
        steps_per_epoch=200,
        batch_size=64,
        validation_data=(X_test, y_test),
        shuffle=True,
        callbacks=[elegy.callbacks.TensorBoard(logdir=logdir)],
    )

    plot_history(history)

    # get random samples
    idxs = np.random.randint(0, 10000, size=(9,))
    x_sample = X_test[idxs]

    # get predictions
    y_pred = model.predict(x=x_sample)

    # plot and save results
    with SummaryWriter(os.path.join(logdir, "val")) as tbwriter:
        figure = plt.figure(figsize=(12, 12))
        for i in range(3):
            for j in range(3):
                k = 3 * i + j
                plt.subplot(3, 3, k + 1)
                plt.title(f"{np.argmax(y_pred[k])}")
                plt.imshow(x_sample[k], cmap="gray")
        tbwriter.add_figure("Predictions", figure, 100)

    plt.show()

    print(
        "\n\n\nMetrics and images can be explored using tensorboard using:",
        f"\n \t\t\t tensorboard --logdir {logdir}",
    )


if __name__ == "__main__":
    typer.run(main)
