from typing import Any, Generator, Mapping, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
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


def loss_fn(y_true, y_pred, params) -> jnp.ndarray:

    l2_loss = elegy.losses.L2Regularization()(params)
    softmax_xent = elegy.losses.SoftmaxCrossentropy()(y_true, y_pred)

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
    train_eval = load_dataset("train", is_training=False, batch_size=100)
    test_eval = load_dataset("test", is_training=False, batch_size=100)

    model = elegy.Model(
        model_fn=net_fn,
        loss=loss_fn,
        metrics=lambda: elegy.metrics.Accuracy(),
        run_eagerly=eager,
    )

    # Train/eval loop.
    for step in range(10001):
        if step > 0 and step % 1000 == 0:
            model.reset_metrics()

            metrics = {}

            for _ in range(10):
                sample = next(train_eval)
                metrics = model.test_on_batch(x=sample, y=sample["label"])

            print(
                f"[Step {step}] - "
                f"Test accuracy: {metrics['accuracy']:.3f} - "
                f"Test Loss: {metrics['loss']:.3f}"
            )

            model.reset_metrics()

        sample = next(train)

        model.train_on_batch(x=sample, y=sample["label"])

    sample = next(test_eval)
    y_pred = model.predict_on_batch(x=sample)

    for i in range(5):
        plt.figure()
        plt.title(f"{np.argmax(y_pred[i])}")
        plt.imshow(sample["image"][i, ..., 0], cmap="gray")

    plt.show()


if __name__ == "__main__":
    typer.run(main)
