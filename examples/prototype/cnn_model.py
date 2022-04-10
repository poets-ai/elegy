import functools
import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import treeo as to
import treex as tx
import typer
from datasets.load import load_dataset
from tqdm import tqdm
from treex import metrics
from treex.utils import _check_rejit

import elegy as eg

Batch = tp.Mapping[str, np.ndarray]
Module = tx.Sequential
Metric = tx.metrics.Accuracy
Logs = tp.Mapping[str, jnp.ndarray]
np.random.seed(420)

M = tp.TypeVar("M", bound="ElegyModule")
C = tp.TypeVar("C", bound="tp.Callable")

# LowLevel (jax) -> Intermediate (pt lightning) -> HighLevel (specialized)


class ElegyModule(to.Tree, to.Immutable, to.Map, to.Copy):
    key: jnp.ndarray = tx.node()

    def __init__(
        self,
        key: tp.Union[jnp.ndarray, int],
        optimizer: optax.GradientTransformation,
        losses: tp.Any,
        metrics: tp.Any,
    ) -> None:
        self.key = tx.Key(key)
        self.module = tx.Sequential(
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
        self.losses_and_metrics = tx.LossesAndMetrics(
            losses=losses,
            metrics=metrics,
        )

    def __call__(self, *args, **kwargs) -> tp.Any:
        return self.module(*args, **kwargs)

    @jax.jit
    @tx.toplevel_mutable
    def init_step(self: M, x: tp.Any) -> M:

        init_key, self.key = jax.random.split(self.key)
        self.module = self.module.init(init_key, x)
        self.optimizer = self.optimizer.init(self.module.parameters())
        self.losses_and_metrics = self.losses_and_metrics.reset()

        return self

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
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> tp.Tuple[jnp.ndarray, "ElegyModule"]:

        if params is not None:
            self.module = self.module.merge(params)

        preds, self.module = self.module.apply(key, x)

        loss, self.losses_and_metrics = self.losses_and_metrics.loss_and_update(
            target=y,
            preds=preds,
        )

        return loss, self

    @jax.jit
    @tx.toplevel_mutable
    def train_step(
        self: M,
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> M:
        print("JITTTTING")
        self.module = self.module.train()

        params = self.module.parameters()
        loss_key, self.key = jax.random.split(self.key)

        grads, self = jax.grad(self.loss_fn, has_aux=True)(params, loss_key, x, y)

        params, self.optimizer = self.optimizer.update(grads, params)
        self.module = self.module.merge(params)

        return self

    @jax.jit
    @tx.toplevel_mutable
    def test_step(
        self: M,
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> M:
        self.module = self.module.eval()
        loss, self = self.loss_fn(None, None, x, y)

        return self

    @jax.jit
    def predict(self, x: jnp.ndarray) -> jnp.ndarray:
        module = self.module.eval()
        return module(x).argmax(axis=1)


# define parameters
def main(
    epochs: int = 5,
    batch_size: int = 64,
    steps_per_epoch: int = -1,
    seed: int = 420,
):

    # load data
    dataset = load_dataset("mnist")
    dataset.set_format("np")
    X_train = np.stack(dataset["train"]["image"])[..., None]
    y_train = dataset["train"]["label"]
    X_test = np.stack(dataset["test"]["image"])[..., None]
    y_test = dataset["test"]["label"]

    # define model
    module = ElegyModule(
        key=seed,
        optimizer=optax.adamw(1e-3),
        losses=tx.losses.Crossentropy(),
        metrics=tx.metrics.Accuracy(),
    )

    module: ElegyModule = module.init_step(X_train[:batch_size])

    print("X_train:", X_train.shape, X_train.dtype)
    print("X_test:", X_test.shape, X_test.dtype)

    model = eg.Model(module)

    model.summary(X_train[:64])

    history = model.fit(
        inputs=X_train,
        labels=y_train,
        epochs=10,
        steps_per_epoch=100,
        batch_size=64,
        validation_data=(X_test, y_test),
        shuffle=True,
        callbacks=[eg.callbacks.TensorBoard("summaries")],
    )

    eg.utils.plot_history(history)

    print(model.evaluate(x=X_test, y=y_test))

    # get random samples
    idxs = np.random.randint(0, 10000, size=(9,))
    x_sample = X_test[idxs]

    # get predictions
    # model = model.local()
    y_pred = model.predict(x=x_sample)

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
