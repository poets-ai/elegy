from typing import Any, Generator, Mapping, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds
import typer

import elegy

OptState = Any
Batch = Mapping[str, np.ndarray]


def load_dataset(
    split: str, *, is_training: bool, batch_size: int,
) -> Generator[Batch, None, None]:
    """Loads the dataset as a generator of batches."""
    ds = tfds.load("mnist:3.*.*", split=split).cache().repeat()
    if is_training:
        ds = ds.shuffle(10 * batch_size, seed=0)
    ds = ds.batch(batch_size)
    return tfds.as_numpy(ds)


def net_fn(image) -> jnp.ndarray:
    """Standard LeNet-300-100 MLP network."""
    image = image.astype(jnp.float32) / 255.0

    mlp = hk.Sequential(
        [
            hk.Flatten(),
            hk.Linear(300),
            jax.nn.relu,
            hk.Linear(100),
            jax.nn.relu,
            hk.Linear(10),
        ]
    )
    return mlp(image)


def accuracy(y_true, y_pred):
    """"""
    return jnp.mean(jnp.argmax(y_pred, axis=-1) == y_true)


def metrics_fn(y_true, y_pred):
    """"""
    return dict(accuracy=accuracy(y_true, y_pred))


def loss_fn(y_true, y_pred, x, params) -> jnp.ndarray:
    """"""
    import matplotlib.pyplot as plt

    label = y_true
    image = x
    for i in range(5):
        plt.figure()
        plt.title(f"{label[i]}")
        plt.imshow(image[i, ..., 0], cmap="gray")
        plt.show()
    labels = jax.nn.one_hot(y_true, 10)

    l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))

    softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(y_pred))
    softmax_xent /= labels.shape[0]

    return softmax_xent + 1e-4 * l2_loss


def main(debug: bool = False, eager: bool = False):

    if debug:
        import debugpy

        print("Waiting for debugger...")
        debugpy.listen(5678)
        debugpy.wait_for_client()

    # We maintain avg_params, the exponential moving average of the "live" params.
    # avg_params is used only for evaluation.
    # For more, see: https://doi.org/10.1137/0330046
    # @jax.jit
    # def ema_update(
    #     avg_params: hk.Params, new_params: hk.Params, epsilon: float = 0.001,
    # ) -> hk.Params:
    #     return jax.tree_multimap(
    #         lambda p1, p2: (1 - epsilon) * p1 + epsilon * p2, avg_params, new_params
    #     )

    # Make datasets.
    train = load_dataset("train", is_training=True, batch_size=64)
    # train_eval = load_dataset("train", is_training=False, batch_size=1000)
    # test_eval = load_dataset("test", is_training=False, batch_size=1000)

    n_iters = 100
    x = []
    y = []
    for _, ex in zip(range(n_iters), train):
        x.append(ex["image"])
        y.append(ex["label"][..., None])
    y = np.vstack(y)
    x = np.vstack(x)
    print(x.shape, y.shape)

    loss_acc = 0
    logs = None

    model = elegy.Model(
        model_fn=net_fn,
        loss=loss_fn,
        metrics=lambda: ("accuracy", accuracy),
        run_eagerly=eager,
    )
    model.fit(x=x, y=y, epochs=10, batch_size=64)
    exit()

    # Train/eval loop.
    for step in range(10001):
        if step > 0 and step % 1000 == 0:
            # Periodically evaluate classification accuracy on train & test sets.
            # train_accuracy = accuracy(avg_params, next(train_eval))
            # test_accuracy = accuracy(avg_params, next(test_eval))
            # train_accuracy, test_accuracy = jax.device_get(
            #     (train_accuracy, test_accuracy)
            # )
            print(
                f"[Step {step}] Train / Test accuracy: "
                f"{logs['accuracy']} - "
                f"Train Loss: {loss_acc/1000:.3f}"
            )
            loss_acc = 0

        sample = next(train)

        logs = model.train_on_batch(x=sample, y=sample["label"])

        loss_acc += logs["loss"]


if __name__ == "__main__":
    typer.run(main)
