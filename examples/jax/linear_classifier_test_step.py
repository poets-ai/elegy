import os
import typing as tp
from datetime import datetime

import dataget
import jax
import jax.numpy as jnp
import numpy as np
import optax
import typer

import elegy as eg

M = tp.TypeVar("M", bound="Model")


class Model(eg.Model):
    w: jnp.ndarray = eg.Parameter.node()
    b: jnp.ndarray = eg.Parameter.node()

    def __init__(
        self,
        features_out: int,
        loss: tp.Any = None,
        metrics: tp.Any = None,
        optimizer=None,
        seed: int = 42,
        eager: bool = False,
    ):
        self.features_out = features_out
        super().__init__(
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            seed=seed,
            eager=eager,
        )

    def init_step(self: M, key: jnp.ndarray, inputs: jnp.ndarray) -> M:
        features_in = np.prod(inputs.shape[1:])

        self.w = jax.random.uniform(
            key,
            shape=[
                features_in,
                self.features_out,
            ],
        )
        self.b = jnp.zeros([self.features_out])

        assert self.optimizer is not None
        self.optimizer = self.optimizer.init(self)

        return self

    def pred_step(self: M, inputs: tp.Any) -> eg.PredStep[M]:
        logits = jnp.dot(inputs, self.w) + self.b
        return logits, self

    def test_step(
        self,
        inputs,
        labels,
    ):
        model = self
        # flatten + scale
        inputs = jnp.reshape(inputs, (inputs.shape[0], -1)) / 255

        # forward
        logits, model = model.pred_step(inputs)

        # crossentropy loss
        target = jax.nn.one_hot(labels["target"], self.features_out)
        loss = jnp.mean(-jnp.sum(target * jax.nn.log_softmax(logits), axis=-1))

        # metrics
        logs = dict(
            acc=jnp.mean(jnp.argmax(logits, axis=-1) == labels["target"]),
            loss=loss,
        )

        return loss, logs, model


def main(
    debug: bool = False,
    eager: bool = False,
    logdir: str = "runs",
    steps_per_epoch: int = 200,
    batch_size: int = 64,
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

    model = Model(
        features_out=10,
        optimizer=optax.adam(1e-3),
        eager=eager,
    )

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


if __name__ == "__main__":
    typer.run(main)
