from tensorboardX.writer import SummaryWriter
from elegy.callbacks.tensorboard import TensorBoard
import os
from datetime import datetime
import typing as tp
from typing import Any, Generator, Mapping, Tuple

import dataget

from flax import linen as nn


import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import typer
import optax

import elegy


Batch = Mapping[str, np.ndarray]
np.random.seed(42)

LATENT_SIZE = 32
MNIST_IMAGE_SHAPE: tp.Sequence[int] = (28, 28)


def kl_divergence(mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    r"""Calculate KL divergence between given and standard gaussian distributions.
    KL(p, q) = H(p, q) - H(p) = -\int p(x)log(q(x))dx - -\int p(x)log(p(x))dx
            = 0.5 * [log(|s2|/|s1|) - 1 + tr(s1/s2) + (m1-m2)^2/s2]
            = 0.5 * [-log(|s1|) - 1 + tr(s1) + m1^2] (if m2 = 0, s2 = 1)
    Args:
        mean: mean vector of the first distribution
        var: diagonal vector of covariance matrix of the first distribution
    Returns:
        A scalar representing KL divergence of the two Gaussian distributions.
    """
    return jnp.mean(
        0.5 * jnp.mean(-jnp.log(std ** 2) - 1.0 + std ** 2 + mean ** 2, axis=-1)
    )


class Encoder(nn.Module):
    """Encoder model."""

    hidden_size: int = 512
    latent_size: int = 128

    @nn.compact
    def __call__(self, x: np.ndarray, rng: elegy.RNGSeq) -> np.ndarray:
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(self.hidden_size)(x)
        x = jax.nn.relu(x)

        mean = nn.Dense(self.latent_size, name="linear_mean")(x)
        log_stddev = nn.Dense(self.latent_size, name="linear_std")(x)
        stddev = jnp.exp(log_stddev)

        # friendly RNG interface: rng.next() == jax.random.split(...)
        z = mean + stddev * jax.random.normal(rng.next(), mean.shape)

        return z, mean, stddev


class Decoder(nn.Module):
    """Decoder model."""

    hidden_size: int = 512
    output_shape: tp.Sequence[int] = MNIST_IMAGE_SHAPE

    @nn.compact
    def __call__(self, z: np.ndarray) -> np.ndarray:
        z = nn.Dense(self.hidden_size)(z)
        z = jax.nn.relu(z)

        logits = nn.Dense(np.prod(self.output_shape))(z)
        logits = jnp.reshape(logits, (-1, *self.output_shape))

        return logits


class VAE(nn.Module):
    hidden_size: int = 512
    latent_size: int = 512
    output_shape: tp.Sequence[int] = MNIST_IMAGE_SHAPE

    @nn.compact
    def __call__(self, x, rng: elegy.RNGSeq):
        z, mean, stddev = Encoder(
            hidden_size=self.hidden_size, latent_size=self.latent_size
        )(x, rng)
        logits = Decoder(hidden_size=self.hidden_size, output_shape=self.output_shape)(
            z
        )
        return logits, mean, stddev

    def generate(self, z):
        return nn.sigmoid(self.decoder(z))


def main(
    steps_per_epoch: int = 200,
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

    X_train, _1, X_test, _2 = dataget.image.mnist(global_cache=True).get()
    # Now binarize data
    X_train = (X_train > 0).astype(jnp.float32)
    X_test = (X_test > 0).astype(jnp.float32)

    print("X_train:", X_train.shape, X_train.dtype)
    print("X_test:", X_test.shape, X_test.dtype)

    vae = VAE(latent_size=LATENT_SIZE)

    # model = VariationalAutoEncoder(latent_size=LATENT_SIZE, optimizer=optax.adam(1e-3))

    def loss(x, y_pred):
        logits, mean, stddev = y_pred
        ce_loss = elegy.losses.binary_crossentropy(x, logits, from_logits=True).mean()
        kl_loss = 2e-1 * kl_divergence(mean, stddev)
        return ce_loss + kl_loss

    model = elegy.Model(
        module=vae,
        loss=loss,
        optimizer=optax.adam(1e-3),
        run_eagerly=eager,
    )

    # Fit with datasets in memory
    history = model.fit(
        x=X_train,
        epochs=epochs,
        batch_size=64,
        steps_per_epoch=steps_per_epoch,
        validation_data=(X_test,),
        shuffle=True,
        callbacks=[TensorBoard(logdir)],
    )

    print(
        "\n\n\nMetrics and images can be explored using tensorboard using:",
        f"\n \t\t\t tensorboard --logdir {logdir}",
    )

    elegy.utils.plot_history(history)

    # get random samples
    idxs = np.random.randint(0, len(X_test), size=(5,))
    x_sample = X_test[idxs]

    # get predictions
    logits, mean, stddev = model.predict(x=x_sample)
    y_pred = jax.nn.sigmoid(logits)

    # plot and save results
    with SummaryWriter(os.path.join(logdir, "val")) as tbwriter:
        figure = plt.figure(figsize=(12, 12))
        for i in range(5):
            plt.subplot(2, 5, i + 1)
            plt.imshow(x_sample[i], cmap="gray")
            plt.subplot(2, 5, 5 + i + 1)
            plt.imshow(y_pred[i], cmap="gray")
        # # tbwriter.add_figure("VAE Example", figure, epochs)

    plt.show()

    # TODO: implement parameter transfer to sample
    # sample
    # model_decoder = elegy.Model(Decoder(latent_size=LATENT_SIZE))

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
