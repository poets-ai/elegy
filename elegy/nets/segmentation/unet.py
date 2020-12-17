import elegy
from elegy import module
import jax, jax.numpy as jnp
import typing as tp


# TODO: backbone
# TODO: mkdocs


class ConvBlock(module.Module):
    def __init__(
        self, channels: int, batchnorm: tp.Optional[bool] = True, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.channels = channels
        self.batchnorm = batchnorm

    def call(self, x: jnp.ndarray):
        x = elegy.nn.Conv2D(
            self.channels,
            (3, 3),
            padding="SAME",
            with_bias=not self.batchnorm,
            dtype=self.dtype,
        )(x)
        x = elegy.nn.BatchNormalization(decay_rate=0.9)(x) if self.batchnorm else x
        x = elegy.to_module(jax.nn.relu)()(x)
        x = elegy.nn.Conv2D(
            self.channels,
            (3, 3),
            padding="SAME",
            with_bias=not self.batchnorm,
            dtype=self.dtype,
        )(x)
        x = elegy.nn.BatchNormalization(decay_rate=0.9)(x) if self.batchnorm else x
        x = elegy.to_module(jax.nn.relu)()(x)
        return x


class DownBlock(module.Module):
    def __init__(self, batchnorm: tp.Optional[bool] = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batchnorm = batchnorm

    def call(self, x):
        x = elegy.nn.MaxPool(
            window_shape=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME"
        )(x)
        x = ConvBlock(
            channels=x.shape[-1] * 2, batchnorm=self.batchnorm, dtype=self.dtype
        )(x)
        return x


class UpBlock(module.Module):
    def __init__(self, batchnorm: tp.Optional[bool] = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batchnorm = batchnorm

    def call(self, x: jnp.ndarray, skipx: jnp.ndarray):
        x = jax.image.resize(x, skipx.shape, method="bilinear")
        x = elegy.nn.Conv2D(skipx.shape[-1], (1, 1), dtype=self.dtype)(x)
        x = jnp.concatenate([x, skipx], axis=-1)
        x = ConvBlock(
            channels=skipx.shape[-1], batchnorm=self.batchnorm, dtype=self.dtype
        )(x)
        return x


class UNet(module.Module):
    """U-Net architecture for image segmentation.
    Original Paper: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
    """

    def __init__(
        self,
        output_channels: int,
        batchnorm: tp.Optional[bool] = True,
        alpha: tp.Optional[float] = 1.0,
        dtype: tp.Optional[tp.Union["float32", "float16"]] = "float32",
        *args,
        **kwargs
    ):
        """Arguments:
        output_channels: Number of output classes.
        batchnorm: Whether or not to use batch normalization after convolutions (Default: True)
        alpha: Width multiplicator to control the number of channels in the inner layers.
               With the default value of 1.0, the first inner layer will use 64 channels.
        dtype: Optional dtype of the convolutions and linear operations,
                either jnp.float32 (default) or jnp.float16 for mixed precision.
        """
        super().__init__(*args, **kwargs)
        self.output_channels = output_channels
        self.batchnorm = batchnorm
        self.alpha = alpha

    def call(self, x: jnp.ndarray):
        x = x0 = ConvBlock(
            channels=int(64 * self.alpha), batchnorm=self.batchnorm, dtype=self.dtype
        )(x)
        x = x1 = DownBlock(dtype=self.dtype)(x)
        x = x2 = DownBlock(dtype=self.dtype)(x)
        x = x3 = DownBlock(dtype=self.dtype)(x)
        x = DownBlock(dtype=self.dtype)(x)
        x = UpBlock(dtype=self.dtype)(x, x3)
        x = UpBlock(dtype=self.dtype)(x, x2)
        x = UpBlock(dtype=self.dtype)(x, x1)
        x = UpBlock(dtype=self.dtype)(x, x0)
        x = elegy.nn.Conv2D(self.output_channels, (1, 1), dtype=self.dtype)(x)
        if x.dtype == jnp.float16:
            to_float32 = lambda x: jnp.asarray(x, jnp.float32)
            x = module.to_module(to_float32)(name="to_float32")(x)
        return x
