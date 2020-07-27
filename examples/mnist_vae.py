from tensorboardX.writer import SummaryWriter
from elegy.callbacks.tensorboard import TensorBoard
import os
from datetime import datetime
import typing as tp
from typing import Any, Generator, Mapping, Tuple

import dataget
import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import typer
from jax.experimental import optix

import elegy
from utils import plot_history

Batch = Mapping[str, np.ndarray]
np.random.seed(42)


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
    return 0.5 * jnp.mean(-jnp.log(std ** 2) - 1.0 + std ** 2 + mean ** 2)


class Encoder(elegy.Module):
    """Encoder model."""

    def __init__(self, hidden_size: int = 512, latent_size: int = 10):
        super().__init__()
        self._hidden_size = hidden_size
        self._latent_size = latent_size

    @hk.transparent
    def call(self, x: np.ndarray) -> np.ndarray:
        x = hk.Flatten()(x)
        x = elegy.nn.Linear(self._hidden_size)(x)
        x = jax.nn.relu(x)

        mean = elegy.nn.Linear(self._latent_size)(x)
        log_stddev = elegy.nn.Linear(self._latent_size)(x)
        stddev = jnp.exp(log_stddev)

        elegy.add_loss("kl_divergence", 0.01 * kl_divergence(mean, stddev))

        z = mean + stddev * jax.random.normal(hk.next_rng_key(), mean.shape)

        return z


class Decoder(elegy.Module):
    """Decoder model."""

    def __init__(
        self,
        hidden_size: int = 512,
        output_shape: tp.Sequence[int] = MNIST_IMAGE_SHAPE,
    ):
        super().__init__()
        self._hidden_size = hidden_size
        self._output_shape = output_shape

    @hk.transparent
    def call(self, z: np.ndarray) -> np.ndarray:
        z = elegy.nn.Linear(self._hidden_size)(z)
        z = jax.nn.relu(z)

        logits = elegy.nn.Linear(jnp.prod(self._output_shape))(z)
        logits = jnp.reshape(logits, (-1, *self._output_shape))

        return logits


class VariationalAutoEncoder(elegy.Module):
    """Main VAE model class, uses Encoder & Decoder under the hood."""

    def __init__(
        self,
        hidden_size: int = 512,
        latent_size: int = 10,
        output_shape: tp.Sequence[int] = MNIST_IMAGE_SHAPE,
    ):
        super().__init__()
        self._hidden_size = hidden_size
        self._latent_size = latent_size
        self._output_shape = output_shape

    @hk.transparent
    def call(self, x: np.ndarray) -> dict:
        x = x.astype(jnp.float32)
        z = Encoder(self._hidden_size, self._latent_size)(x)

        logits = Decoder(self._hidden_size, self._output_shape)(z)

        p = jax.nn.sigmoid(logits)
        image = jax.random.bernoulli(hk.next_rng_key(), p)

        return dict(image=image, logits=logits)


class BinaryCrossEntropy(elegy.Loss):
    def call(self, x: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate binary (logistic) cross-entropy from distribution logits.
    Args:
        x: input variable tensor, must be of same shape as logits
        logits: log odds of a Bernoulli distribution, i.e. log(p/(1-p))
    Returns:
        A scalar representing binary CE for the given Bernoulli distribution.
    """
        if x.shape != y_pred.shape:
            raise ValueError(
                "inputs x and y_pred must be of the same shape"
                f"got {x.shape} and {y_pred.shape}"
            )

        x = jnp.reshape(x, (x.shape[0], -1))
        y_pred = jnp.reshape(y_pred, (y_pred.shape[0], -1))

        return -jnp.mean(x * y_pred - jnp.logaddexp(0.0, y_pred), axis=-1)


def main(debug: bool = False, eager: bool = False, logdir: str = "runs"):

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

    model = elegy.Model(
        module=VariationalAutoEncoder.defer(),
        loss=[BinaryCrossEntropy(on="logits")],
        optimizer=optix.adam(1e-3),
        run_eagerly=eager,
    )

    model.summary(X_train[:64])

    epochs = 50

    # Fit with datasets in memory
    history = model.fit(
        x=X_train,
        epochs=epochs,
        batch_size=64,
        steps_per_epoch=200,
        validation_data=(X_test,),
        shuffle=True,
        callbacks=[TensorBoard(logdir)],
    )
    plot_history(history)

    # get random samples
    idxs = np.random.randint(0, len(X_test), size=(5,))
    x_sample = X_test[idxs]

    # get predictions
    y_pred = model.predict(x=x_sample)

    # plot and save results
    with SummaryWriter(os.path.join(logdir, "val")) as tbwriter:
        figure = plt.figure(figsize=(12, 12))
        for i in range(5):
            plt.subplot(2, 5, i + 1)
            plt.imshow(x_sample[i], cmap="gray")
            plt.subplot(2, 5, 5 + i + 1)
            plt.imshow(y_pred["image"][i], cmap="gray")
        tbwriter.add_figure("VAE Example", figure, epochs)

    plt.show()

    print(
        "\n\n\nMetrics and images can be explored using tensorboard using:",
        f"\n \t\t\t tensorboard --logdir {logdir}",
    )


if __name__ == "__main__":
    typer.run(main)
