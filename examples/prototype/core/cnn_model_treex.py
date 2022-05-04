import functools
import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
import jax_metrics as jm
import matplotlib.pyplot as plt
import numpy as np
import optax
import treeo as to
import treex as tx
import typer
from datasets.load import load_dataset

import elegy as eg

Batch = tp.Mapping[str, np.ndarray]
Module = tx.Sequential
Metric = jm.metrics.Accuracy
Logs = tp.Mapping[str, jnp.ndarray]
np.random.seed(420)

M = tp.TypeVar("M", bound="ElegyModule")
C = tp.TypeVar("C", bound="tp.Callable")


# LowLevel (jax) -> Intermediate (pt lightning) -> HighLevel (specialized)


def set_training(**fields: bool):
    def decorator(f: C) -> C:
        @functools.wraps(f)
        def wrapper(self: "ElegyModule", *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
            self = self.replace(
                **{
                    field: getattr(self, field).train(mode)
                    for field, mode in fields.items()
                }
            )

            return f(self, *args, **kwargs)

        return wrapper

    return decorator


class ElegyModule(eg.Module):
    key: tp.Optional[jnp.ndarray] = eg.node()

    def __init__(
        self,
        optimizer: optax.GradientTransformation,
        losses: tp.Any,
        metrics: tp.Any,
    ) -> None:
        super().__init__()
        self.key = None
        self.module = tx.Sequential(
            lambda x: x.astype(jnp.float32) / 255.0,
            tx.Conv(32, [3, 3], strides=[2, 2]),
            tx.BatchNorm(),
            tx.Dropout(0.05),
            jax.nn.relu,
            tx.Conv(64, [3, 3], strides=[2, 2]),
            tx.BatchNorm(),
            tx.Dropout(0.1),
            jax.nn.relu,
            tx.Conv(128, [3, 3], strides=[2, 2]),
            partial(jnp.mean, axis=(1, 2)),
            tx.Linear(10),
        )
        self.optimizer = tx.Optimizer(optimizer)
        self.losses_and_metrics = jm.LossesAndMetrics(
            losses=losses,
            metrics=metrics,
        )

    def __call__(self, *args, **kwargs) -> tp.Any:
        return self.module(*args, **kwargs)

    @set_training(module=True)
    @jax.jit
    @tx.toplevel_mutable
    def init_step(
        self: M,
        key: jnp.ndarray,
        batch: tp.Union[jnp.ndarray, tp.Tuple[jnp.ndarray, jnp.ndarray]],
    ) -> M:
        if isinstance(batch, tuple):
            inputs, _ = batch
        else:
            inputs = batch

        init_key, self.key = jax.random.split(key)
        self.module = self.module.init(key=init_key)(inputs)
        self.optimizer = self.optimizer.init(self.module.parameters())
        self.losses_and_metrics = self.losses_and_metrics.init()

        return self

    @set_training(module=True)
    @jax.jit
    @tx.toplevel_mutable
    def reset_step(self: M) -> M:
        self.losses_and_metrics = self.losses_and_metrics.reset()
        return self

    @tx.toplevel_mutable
    def loss_fn(
        self: "ElegyModule",
        params: tp.Optional[Module],
        key: tp.Optional[jnp.ndarray],
        inputs: jnp.ndarray,
        labels: jnp.ndarray,
    ) -> tp.Tuple[jnp.ndarray, "ElegyModule"]:

        if params is not None:
            self.module = self.module.merge(params)

        preds, self.module = self.module.apply(key=key)(inputs)

        loss, self.losses_and_metrics = self.losses_and_metrics.loss_and_update(
            target=labels,
            preds=preds,
        )

        return loss, self

    @set_training(module=True)
    @jax.jit
    @tx.toplevel_mutable
    def train_step(
        self: M,
        batch: tp.Tuple[jnp.ndarray, jnp.ndarray],
        batch_idx: int,
        epoch_idx: int,
    ) -> tp.Tuple[Logs, M]:
        print("JITTTTING")

        assert self.key is not None

        inputs, labels = batch

        params = self.module.parameters()
        loss_key, self.key = jax.random.split(self.key)

        grads, self = jax.grad(self.loss_fn, has_aux=True)(
            params, loss_key, inputs, labels
        )

        params, self.optimizer = self.optimizer.update(grads, params)
        self.module = self.module.merge(params)

        logs = self.losses_and_metrics.compute_logs()

        return logs, self

    @set_training(module=False)
    @jax.jit
    @tx.toplevel_mutable
    def test_step(
        self: M,
        batch: tp.Tuple[jnp.ndarray, jnp.ndarray],
        batch_idx: int,
    ) -> tp.Tuple[Logs, M]:
        inputs, labels = batch

        loss, self = self.loss_fn(None, None, inputs, labels)

        logs = self.losses_and_metrics.compute_logs()

        return logs, self

    @set_training(module=False)
    @jax.jit
    def predict_step(
        self: M,
        batch: jnp.ndarray,
        batch_idx: int,
    ) -> tp.Tuple[tp.Any, M]:
        inputs = batch

        module = self.module.eval()
        outputs = module(inputs).argmax(axis=1)

        return outputs, self

    def tabulate(
        self,
        inputs,
        summary_depth: int = 2,
    ) -> str:
        return self.module.tabulate(inputs, summary_depth=summary_depth)


# define parameters
def main(
    epochs: int = 10,
    batch_size: int = 32,
    steps_per_epoch: tp.Optional[int] = None,
    seed: int = 420,
):

    # load data
    dataset = load_dataset("mnist")
    dataset.set_format("np")
    X_train = np.stack(dataset["train"]["image"])[..., None]
    y_train = dataset["train"]["label"]
    X_test = np.stack(dataset["test"]["image"])[..., None]
    y_test = dataset["test"]["label"]

    print("X_train:", X_train.shape, X_train.dtype)
    print("X_test:", X_test.shape, X_test.dtype)

    model = eg.Model(
        ElegyModule(
            optimizer=optax.adamw(1e-3),
            losses=jm.losses.Crossentropy(),
            metrics=jm.metrics.Accuracy(),
        )
    )

    model.summary(X_train[:64])

    history = model.fit(
        inputs=X_train,
        labels=y_train,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        shuffle=True,
        callbacks=[eg.callbacks.TensorBoard("summaries/prototype")],
    )

    eg.utils.plot_history(history)

    print(model.evaluate(X_test, y_test))

    # get random samples
    idxs = np.random.randint(0, 10000, size=(9,))
    x_sample = X_test[idxs]

    # get predictions
    y_pred = model.predict(x_sample)

    # plot results
    figure = plt.figure(figsize=(12, 12))
    for i in range(3):
        for j in range(3):
            k = 3 * i + j
            plt.subplot(3, 3, k + 1)

            plt.title(f"{y_pred[k]}")
            plt.imshow(x_sample[k], cmap="gray")

    plt.show()


if __name__ == "__main__":

    typer.run(main)
