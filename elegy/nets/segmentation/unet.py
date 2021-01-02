import elegy
from elegy.module import Module, to_module
import jax, jax.numpy as jnp
import numpy as np
import typing as tp


# TODO: mkdocs


class ConvBlock(Module):
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


class DownBlock(Module):
    def __init__(
        self, channels: int, batchnorm: tp.Optional[bool] = True, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.channels = channels
        self.batchnorm = batchnorm

    def call(self, x):
        x = elegy.nn.MaxPool(
            window_shape=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME"
        )(x)
        x = ConvBlock(self.channels, batchnorm=self.batchnorm, dtype=self.dtype)(x)
        return x


class UpBlock(Module):
    def __init__(
        self, channels: int, batchnorm: tp.Optional[bool] = True, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.channels = channels
        self.batchnorm = batchnorm

    def call(self, x: jnp.ndarray, skipx: jnp.ndarray):
        x = jax.image.resize(x, skipx.shape, method="bilinear")
        x = elegy.nn.Conv2D(self.channels, (1, 1), dtype=self.dtype)(x)
        x = jnp.concatenate([x, skipx], axis=-1)
        x = ConvBlock(self.channels, batchnorm=self.batchnorm, dtype=self.dtype)(x)
        return x


class UNet(Module):
    """U-Net architecture for image segmentation.
    Original Paper: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
    """

    def __init__(
        self,
        output_channels: int,
        encoder: tp.Union[Module, tp.List[int]] = [64, 128, 256, 512, 1024],
        decoder: tp.List[int] = [512, 256, 128, 64],
        batchnorm: tp.Optional[bool] = True,
        dtype: tp.Optional[tp.Union["float32", "float16"]] = "float32",
        *args,
        **kwargs
    ):
        """Arguments:
        output_channels: Number of output classes.
        encoder: TODO
        decoder: TODO
        batchnorm: Whether or not to use batch normalization after convolutions (Default: True)
        dtype: Optional dtype of the convolutions and linear operations,
                either jnp.float32 (default) or jnp.float16 for mixed precision.
        """
        super().__init__(*args, dtype=dtype, **kwargs)
        self.output_channels = output_channels
        self.batchnorm = batchnorm

        if isinstance(encoder, tp.List):
            assert len(encoder) == len(decoder) + 1
        self.encoder, self.decoder = encoder, decoder

    def call(self, x: jnp.ndarray):
        if isinstance(self.encoder, tp.List):
            skip_x = [
                ConvBlock(self.encoder[0], batchnorm=self.batchnorm, dtype=self.dtype)(
                    x
                )
            ]
            for channels in self.encoder[1:]:
                skip_x += [
                    DownBlock(channels, self.batchnorm, dtype=self.dtype)(skip_x[-1])
                ]
        else:
            skip_x = self.encoder(x)
            assert isinstance(skip_x, (tp.List, tp.Tuple))
            assert len(skip_x) == len(self.decoder) + 1

        x = skip_x[-1]
        for channels, skip in zip(self.decoder, reversed(skip_x[:-1])):
            x = UpBlock(channels, self.batchnorm, dtype=self.dtype)(x, skip)

        x = elegy.nn.Conv2D(self.output_channels, (1, 1), dtype=self.dtype)(x)
        if x.dtype == jnp.float16:
            to_float32 = lambda x: jnp.asarray(x, jnp.float32)
            x = to_module(to_float32)(name="to_float32")(x)
        return x


class UNet_R18(UNet):
    """U-Net with a ResNet18 backbone"""

    def __init__(
        self,
        output_channels: int,
        *args,
        backbone_weights="imagenet",
        dtype: tp.Optional[tp.Union["float32", "float16"]] = "float32",
        **kwargs
    ):
        r18 = elegy.nets.resnet.ResNet18(weights=backbone_weights, dtype=dtype)
        r18encoder = r18.slice(
            start_module=None,
            end_module=[
                "/input",
                "/relu",
                "/res_net_block_1",
                "/res_net_block_3",
                "/res_net_block_5",
                "/res_net_block_7",
            ],
            sample_input=np.zeros([1, 128, 128, 3]),  # does sample_input size matter?
        )
        super().__init__(
            output_channels,
            encoder=r18encoder,
            decoder=[256, 128, 64, 32, 16],
            dtype=dtype,
        )


class UNet_R50(UNet):
    """U-Net with a ResNet50 backbone"""

    def __init__(
        self,
        output_channels: int,
        *args,
        backbone_weights="imagenet",
        dtype: tp.Optional[tp.Union["float32", "float16"]] = "float32",
        **kwargs
    ):
        r50 = elegy.nets.resnet.ResNet50(weights=backbone_weights, dtype=dtype)
        r50encoder = r50.slice(
            start_module=None,
            end_module=[
                "/input",
                "/relu",
                "/bottleneck_res_net_block_2",
                "/bottleneck_res_net_block_6",
                "/bottleneck_res_net_block_12",
                "/bottleneck_res_net_block_15",
            ],
            sample_input=np.zeros([1, 128, 128, 3]),  # does sample_input size matter?
        )
        super().__init__(
            output_channels,
            encoder=r50encoder,
            decoder=[512, 256, 128, 64, 32],
            dtype=dtype,
        )
