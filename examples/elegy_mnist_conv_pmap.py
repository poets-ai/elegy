"""
In this simple example we just explore distributed training. Evaluation and inference are done on a single device
but the strategy used can be applied to these other scenarious as well with the following modifications:

If `train_step`, `test_step`, `pred_step` were to use `pmap`, then `pmap` would be called more than once for training or evaluation,
to get around this issue the idea would be to override `call_train_step`, `call_test_step` and `call_pred_step` instead since they behave like
entrypoints and don't depend each other.
"""

import os
from datetime import datetime
from typing import Any, Generator, Mapping, Tuple

import dataget
import einops
import elegy
import jax
import jax.numpy as jnp
import jax.tools.colab_tpu
import matplotlib.pyplot as plt
import numpy as np
import optax
import typer
from elegy import utils
from elegy.callbacks.tensorboard import TensorBoard
from tensorboardX.writer import SummaryWriter


def replicate(x):
    return jax.tree_map(
        lambda a: einops.repeat(a, "... -> device ...", device=jax.device_count()),
        x,
    )


class ReplicateStates(elegy.callbacks.Callback):
    def __init__(self):
        self.replicated = False

    def on_train_begin(self, *args, **kwargs):
        if not self.replicated:
            self.replicate()
            self.replicated = True

    def on_train_end(self, *args, **kwargs):
        self.dereplicate()
        self.replicated = False

    def on_test_begin(self, *args, **kwargs):
        if self.replicated:
            self.dereplicate()

    def on_test_end(self, *args, **kwargs):
        if self.replicated:
            self.replicate()

    def replicate(self):
        self.model.states = replicate(self.model.states)
        self.model.initial_states = replicate(self.model.initial_states)

    def dereplicate(self):
        self.model.states = jax.tree_map(lambda x: x[0], self.model.states)
        self.model.initial_states = jax.tree_map(
            lambda x: x[0], self.model.initial_states
        )


class DistributedModel(elegy.Model):
    def jit_step(self):
        super().jit_step()

        self.train_step_pmap = jax.pmap(
            super().train_step,
            static_broadcasted_argnums=[5, 6],
            axis_name="device",
        )
        self.call_train_step_jit = self.call_train_step

        self.jitted_members |= {"train_step_pmap"}

    def grad_step(self, *args, **kwargs):
        loss, logs, states, grads = super().grad_step(*args, **kwargs)

        grads = jax.lax.psum(grads, axis_name="device")

        return loss, logs, states, grads

    # here we override train_step instead of train_step
    def call_train_step(
        self,
        x,
        y_true,
        sample_weight,
        class_weight,
        states,
        initializing,
        training,
    ):
        x = jax.tree_map(
            lambda a: einops.rearrange(
                a, "(device batch) ... -> device batch ...", device=jax.device_count()
            ),
            x,
        )
        y_true = jax.tree_map(
            lambda a: einops.rearrange(
                a, "(device batch) ... -> device batch ...", device=jax.device_count()
            ),
            y_true,
        )

        logs, states = self.train_step_pmap(
            x,
            y_true,
            sample_weight,
            class_weight,
            states,
            initializing,
            training,
        )

        logs = {name: value[0] for name, value in logs.items()}

        return logs, states


def main(
    debug: bool = False,
    eager: bool = False,
    logdir: str = "runs",
    steps_per_epoch: int = 200,
    epochs: int = 100,
    batch_size: int = 64,
    use_tpu: bool = False,
):

    if debug:
        import debugpy

        print("Waiting for debugger...")
        debugpy.listen(5678)
        debugpy.wait_for_client()

    if use_tpu:
        print("Setting up TPU...")
        jax.tools.colab_tpu.setup_tpu()

    batch_size *= jax.device_count()
    lr = 1e-3 * jax.device_count()

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    logdir = os.path.join(logdir, current_time)

    X_train, y_train, X_test, y_test = dataget.image.mnist(global_cache=True).get()

    X_train = X_train[..., None]
    X_test = X_test[..., None]

    print("X_train:", X_train.shape, X_train.dtype)
    print("y_train:", y_train.shape, y_train.dtype)
    print("X_test:", X_test.shape, X_test.dtype)
    print("y_test:", y_test.shape, y_test.dtype)
    print("Batch size:", batch_size)
    print("Learning Rate:", lr)

    class CNN(elegy.Module):
        def call(self, image: jnp.ndarray, training: bool):
            @elegy.to_module
            def ConvBlock(x, units, kernel, stride=1):
                x = elegy.nn.Conv2D(units, kernel, stride=stride, padding="same")(x)
                x = elegy.nn.BatchNormalization()(x, training)
                x = elegy.nn.Dropout(0.2)(x, training)
                return jax.nn.relu(x)

            x: np.ndarray = image.astype(jnp.float32) / 255.0

            # base
            x = ConvBlock()(x, 32, [3, 3])
            x = ConvBlock()(x, 64, [3, 3], stride=2)
            x = ConvBlock()(x, 64, [3, 3], stride=2)
            x = ConvBlock()(x, 128, [3, 3], stride=2)

            # GlobalAveragePooling2D
            x = jnp.mean(x, axis=[1, 2])

            # 1x1 Conv
            x = elegy.nn.Linear(10)(x)

            return x

    model = DistributedModel(
        module=CNN(),
        loss=elegy.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=elegy.metrics.SparseCategoricalAccuracy(),
        optimizer=optax.adam(lr),
        run_eagerly=eager,
    )

    # show summary
    model.summary(X_train[:64])

    history = model.fit(
        x=X_train,
        y=y_train,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        shuffle=True,
        callbacks=[TensorBoard(logdir=logdir), ReplicateStates()],
    )

    elegy.utils.plot_history(history)

    model.save("models/conv")
    model = elegy.load("models/conv")

    # get random samples
    idxs = np.random.randint(0, 10000, size=(9,))
    x_sample = X_test[idxs]

    # get predictions
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
