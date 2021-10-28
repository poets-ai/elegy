import os
import typing as tp
from datetime import datetime
from typing import Any, Generator, Mapping, Tuple

import einops
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import typer
from datasets.load import load_dataset
from tensorboardX.writer import SummaryWriter

import elegy as eg
from elegy import utils

Batch = Mapping[str, np.ndarray]
np.random.seed(42)

LATENT_SIZE = 32
MNIST_IMAGE_SHAPE: tp.Sequence[int] = (28, 28)


class KLDivergence(eg.Loss):
    def call(self, mean: jnp.ndarray, std: jnp.ndarray) -> jnp.ndarray:
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
        return 0.5 * jnp.mean(-jnp.log(std ** 2) - 1.0 + std ** 2 + mean ** 2, axis=-1)


class Encoder(eg.Module):
    """Encoder model."""

    kl_loss: jnp.ndarray = eg.LossLog.node()

    def __init__(self, hidden_size: int = 512, latent_size: int = 128):
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.kl_loss = jnp.array(0.0, dtype=jnp.float32)
        self.next_key = eg.KeySeq()

    @eg.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = einops.rearrange(x, "batch height width -> batch (height width)")
        x = eg.Linear(self.hidden_size)(x)
        x = jax.nn.relu(x)

        mean = eg.Linear(self.latent_size, name="linear_mean")(x)
        log_stddev = eg.Linear(self.latent_size, name="linear_std")(x)
        stddev = jnp.exp(log_stddev)

        self.kl_loss = KLDivergence(weight=2e-1)(mean=mean, std=stddev)

        z = mean + stddev * jax.random.normal(self.next_key(), mean.shape)

        return z


class Decoder(eg.Module):
    """Decoder model."""

    def __init__(
        self,
        hidden_size: int = 512,
        output_shape: tp.Sequence[int] = MNIST_IMAGE_SHAPE,
    ):
        super().__init__()
        self.output_shape = output_shape
        self.hidden_size = hidden_size

    @eg.compact
    def __call__(self, z: jnp.ndarray) -> np.ndarray:
        z = eg.Linear(self.hidden_size)(z)
        z = jax.nn.relu(z)

        logits = eg.Linear(np.prod(self.output_shape))(z)
        logits = jnp.reshape(logits, (-1, *self.output_shape))

        return logits


class VariationalAutoEncoder(eg.Module):
    """Main VAE model class, uses Encoder & Decoder under the hood."""

    encoder: Encoder
    decoder: Decoder

    def __init__(
        self,
        hidden_size: int = 512,
        latent_size: int = 32,
        output_shape: tp.Sequence[int] = MNIST_IMAGE_SHAPE,
    ):
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.output_shape = output_shape

    @eg.compact
    def __call__(self, x: jnp.ndarray) -> dict:
        next_key = eg.KeySeq()
        x = x.astype(jnp.float32)

        z = Encoder(self.hidden_size, self.latent_size)(x)
        logits = Decoder(self.hidden_size, self.output_shape)(z)

        p = jax.nn.sigmoid(logits)
        image = jax.random.bernoulli(next_key(), p)

        return dict(image=image, logits=logits, det_image=p)


class BinaryCrossEntropy(eg.losses.Crossentropy):
    def __init__(self, **kwargs):
        super().__init__(binary=True, **kwargs)

    def call(self, inputs: jnp.ndarray, preds: jnp.ndarray) -> jnp.ndarray:
        return super().call(target=inputs, preds=preds)


def main(
    steps_per_epoch: tp.Optional[int] = None,
    batch_size: int = 32,
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

    dataset = load_dataset("fashion_mnist")
    dataset.set_format("np")
    X_train = np.array(dataset["train"]["image"], dtype=np.uint8)
    X_test = np.array(dataset["test"]["image"], dtype=np.uint8)

    # Now binarize data
    X_train = (X_train / 255.0).astype(jnp.float32)
    X_test = (X_test / 255.0).astype(jnp.float32)

    print("X_train:", X_train.shape, X_train.dtype)
    print("X_test:", X_test.shape, X_test.dtype)

    model = eg.Model(
        module=VariationalAutoEncoder(latent_size=LATENT_SIZE),
        loss=[BinaryCrossEntropy(from_logits=True, on="logits")],
        optimizer=optax.adam(1e-3),
        eager=eager,
    )
    assert model.module is not None

    model.summary(X_train[:64])

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
    y_pred = model.predict(x=x_sample)

    # plot and save results
    with SummaryWriter(os.path.join(logdir, "val")) as tbwriter:
        figure = plt.figure(figsize=(12, 12))
        for i in range(5):
            plt.subplot(2, 5, i + 1)
            plt.imshow(x_sample[i], cmap="gray")
            plt.subplot(2, 5, 5 + i + 1)
            plt.imshow(y_pred["det_image"][i], cmap="gray")
        # tbwriter.add_figure("VAE Example", figure, epochs)

    # sample
    model_decoder = eg.Model(model.module.decoder)

    z_samples = np.random.normal(size=(12, LATENT_SIZE))
    samples = model_decoder.predict(z_samples)
    samples = jax.nn.sigmoid(samples)

    # plot and save results
    # with SummaryWriter(os.path.join(logdir, "val")) as tbwriter:
    figure = plt.figure(figsize=(5, 12))
    plt.title("Generative Samples")
    for i in range(5):
        plt.subplot(2, 5, 2 * i + 1)
        plt.imshow(samples[i], cmap="gray")
        plt.subplot(2, 5, 2 * i + 2)
        plt.imshow(samples[i + 1], cmap="gray")
    # # tbwriter.add_figure("VAE Generative Example", figure, epochs)

    plt.show()


if __name__ == "__main__":
    typer.run(main)
