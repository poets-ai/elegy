import os
import typing as tp
from datetime import datetime
from typing import Any, Generator, Mapping, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import typer
from datasets.load import load_dataset
from tensorboardX.writer import SummaryWriter

import elegy as eg

Batch = Mapping[str, jnp.ndarray]
np.random.seed(42)

LATENT_SIZE = 32
MNIST_IMAGE_SHAPE: tp.Sequence[int] = (28, 28)


class Encoder(hk.Module):
    """Encoder model."""

    def __init__(self, hidden_size: int = 512, latent_size: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size

    def __call__(
        self, x: jnp.ndarray
    ) -> tp.Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        x = x.reshape((x.shape[0], -1))  # flatten
        x = hk.Linear(self.hidden_size)(x)
        x = jax.nn.relu(x)

        mean = hk.Linear(self.latent_size, name="linear_mean")(x)
        log_stddev = hk.Linear(self.latent_size, name="linear_std")(x)
        stddev = jnp.exp(log_stddev)

        z = mean + stddev * jax.random.normal(hk.next_rng_key(), mean.shape)

        return z, mean, stddev


class Decoder(hk.Module):
    """Decoder model."""

    def __init__(
        self, hidden_size: int = 512, output_shape: tp.Sequence[int] = MNIST_IMAGE_SHAPE
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_shape = output_shape

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        z = hk.Linear(self.hidden_size)(z)
        z = jax.nn.relu(z)

        logits = hk.Linear(np.prod(self.output_shape))(z)
        logits = jnp.reshape(logits, (-1, *self.output_shape))

        return logits


class VAE(hk.Module):
    def __init__(
        self,
        hidden_size: int = 512,
        latent_size: int = 512,
        output_shape: tp.Sequence[int] = MNIST_IMAGE_SHAPE,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.output_shape = output_shape

        self.encoder = Encoder(
            hidden_size=self.hidden_size, latent_size=self.latent_size
        )
        self.decoder = Decoder(
            hidden_size=self.hidden_size, output_shape=self.output_shape
        )

    def __call__(self, x):
        z, mean, std = self.encoder(x)
        logits = self.decoder(z)
        return dict(logits=logits, mean=mean, std=std)

    def generate(self, z):
        return jax.nn.sigmoid(self.decoder(z))


class KL(eg.Loss):
    def call(self, preds) -> jnp.ndarray:
        mean = preds["mean"]
        std = preds["std"]

        return 0.5 * jnp.mean(-jnp.log(std**2) - 1.0 + std**2 + mean**2, axis=-1)


class BinaryCrossEntropy(eg.losses.Crossentropy):
    def __init__(self, **kwargs):
        super().__init__(binary=True, **kwargs)

    def call(self, inputs: jnp.ndarray, preds: jnp.ndarray) -> jnp.ndarray:
        return super().call(target=inputs, preds=preds)


def main(
    steps_per_epoch: int = 200,
    batch_size: int = 64,
    epochs: int = 50,
    debug: bool = False,
    eager: bool = False,
    logdir: str = "runs",
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
    X_train = np.stack(dataset["train"]["image"])
    X_test = np.stack(dataset["test"]["image"])
    # Now binarize data
    X_train = (X_train > 0).astype(jnp.float32)
    X_test = (X_test > 0).astype(jnp.float32)

    print("X_train:", X_train.shape, X_train.dtype)
    print("X_test:", X_test.shape, X_test.dtype)

    def forward(x: jnp.ndarray):
        return VAE(latent_size=LATENT_SIZE)(x)

    model = eg.Model(
        module=hk.transform_with_state(forward),
        loss=[
            BinaryCrossEntropy(on="logits"),
            KL(weight=0.1),
        ],
        optimizer=optax.adam(1e-3),
        eager=eager,
    )

    model.summary(X_train[:batch_size])

    # Fit with datasets in memory
    history = model.fit(
        inputs=X_train,
        epochs=epochs,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        validation_data=(X_test,),
        shuffle=True,
        callbacks=[eg.callbacks.TensorBoard(logdir)],
    )

    print(
        "\n\n\nMetrics and images can be explored using tensorboard using:",
        f"\n \t\t\t tensorboard --logdir {logdir}",
    )

    eg.utils.plot_history(history)

    # get random samples
    idxs = np.random.randint(0, len(X_test), size=(5,))
    x_sample = X_test[idxs]

    # get predictions
    preds = model.predict(x=x_sample)
    y_pred = jax.nn.sigmoid(preds["logits"])

    # plot and save results
    figure = plt.figure(figsize=(12, 12))
    for i in range(5):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_sample[i], cmap="gray")
        plt.subplot(2, 5, 5 + i + 1)
        plt.imshow(y_pred[i], cmap="gray")

    plt.show()

    # TODO: implement parameter transfer to sample
    # sample
    # model_decoder = eg.Model(Decoder(latent_size=LATENT_SIZE))

    # z_samples = np.random.normal(size=(12, LATENT_SIZE))
    # samples = model_decoder.predict(z_samples)
    # samples = jax.nn.sigmoid(samples)

    # # plot and save results
    # # with SummaryWriter(os.path.join(logdir, "val")) as tbwriter:
    # figure = plt.figure(figsize=(5, 12))
    # plt.title("Generative Samples")
    # for i in range(5):
    #     plt.subplot(2, 5, 2 * i + 1)
    #     plt.imshow(samples[i], cmap="gray")
    #     plt.subplot(2, 5, 2 * i + 2)
    #     plt.imshow(samples[i + 1], cmap="gray")
    # # # tbwriter.add_figure("VAE Generative Example", figure, epochs)

    # plt.show()


if __name__ == "__main__":
    typer.run(main)
