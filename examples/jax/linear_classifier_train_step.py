import os
import typing as tp
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
import optax
import typer
from datasets.load import load_dataset

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
        self: M,
        inputs,
        labels,
    ) -> eg.TestStep[M]:
        model: M = self
        # flatten + scale
        inputs = jnp.reshape(inputs, (inputs.shape[0], -1)) / 255

        # forward
        logits, model = model.pred_step(inputs)

        # crossentropy loss
        target = jax.nn.one_hot(labels["target"], self.features_out)
        loss = optax.softmax_cross_entropy(logits, target).mean()

        # metrics
        logs = dict(
            acc=jnp.mean(jnp.argmax(logits, axis=-1) == labels["target"]),
            loss=loss,
        )

        return loss, logs, model

    @staticmethod
    def loss_fn(params: M, model: M, inputs, labels) -> eg.LossStep[M]:
        model = model.merge(params)
        loss, logs, model = model.test_step(inputs, labels)
        return loss, (logs, model)

    def train_step(self: M, inputs, labels) -> eg.TrainStep[M]:
        model: M = self

        params = model.parameters()
        # train
        grads, (logs, model) = jax.grad(Model.loss_fn, has_aux=True)(
            params,
            model,
            inputs,
            labels,
        )

        assert model.optimizer is not None

        params = model.optimizer.update(grads, params)
        model = model.merge(params)

        return logs, model


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

    dataset = load_dataset("mnist")
    dataset.set_format("np")
    X_train = dataset["train"]["image"]
    y_train = dataset["train"]["label"]
    X_test = dataset["test"]["image"]
    y_test = dataset["test"]["label"]

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
