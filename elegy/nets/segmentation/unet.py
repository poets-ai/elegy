import elegy
from elegy.module import Module, to_module, load_pretrained_weights
import jax, jax.numpy as jnp
import numpy as np
import typing as tp


__all__ = [
    "UNet",
    "UNet_R18",
    "UNet_R50",
]


PRETRAINED_URLS = {
    "coco": {
        "UNet_R18": {
            "url": "https://github.com/poets-ai/elegy-assets/releases/download/unet_r18_coco_rev0/UNet_R18_COCO_weights_rev0.pkl",
            "sha256": "50d99d96ccca475fc80d068cdb4457566fdd0ccde50bdca95c179905020b9176",
        },
        "UNet_R50": {
            "url": "https://github.com/poets-ai/elegy-assets/releases/download/unet_r50_coco_rev0/UNet_R50_COCO_weights_rev0.pkl",
            "sha256": "c435c000a517f3c249de20d0623b76e8aa7c65358fd65612f42e9bb729c4a23e",
        },
    },
}


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
        """
        Arguments:
            output_channels: Number of output classes.
            encoder: Either a list of integers representing the number of output channels
                     for each block of the encoder or a module that directly computes the encoder outputs.
            decoder: A list of integers representing the number of output channels for each block of the decoder.
                     Must have one block less than the outputs of the encoder.
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
    """U-Net with a ResNet18 backbone."""

    def __init__(
        self,
        output_channels: int = 73,
        weights: tp.Union[None, str, "coco"] = None,
        backbone_weights: tp.Union[None, str, "imagenet"] = None,
        dtype: tp.Optional[tp.Union["float32", "float16"]] = "float32",
        **kwargs
    ):
        if weights is not None:
            backbone_weights = None

        r18 = elegy.nets.resnet.ResNet18(weights=backbone_weights, dtype=dtype)
        _x = np.zeros([1, 64, 64, 3])
        elegy.Model(r18, run_eagerly=True).predict(_x)
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
            sample_input=_x,
        )
        if weights == "coco":
            # models pretrained on coco have a fixed number of 73 output channels
            output_channels = 73
        super().__init__(
            output_channels,
            encoder=r18encoder,
            decoder=[256, 128, 64, 32, 16],
            dtype=dtype,
        )

        # manually injecting encoder into submodules, usually this happens after __init__
        # XXX: ugly ugly
        self._submodules.append("encoder")
        if weights is not None:
            load_pretrained_weights(self, weights, PRETRAINED_URLS)


class UNet_R50(UNet):
    """U-Net with a ResNet50 backbone."""

    def __init__(
        self,
        output_channels: int = 73,
        weights: tp.Union[None, str, "coco"] = None,
        backbone_weights: tp.Union[None, str, "imagenet"] = None,
        dtype: tp.Optional[tp.Union["float32", "float16"]] = "float32",
        **kwargs
    ):
        if weights is not None:
            backbone_weights = None

        r50 = elegy.nets.resnet.ResNet50(weights=backbone_weights, dtype=dtype)
        _x = np.zeros([1, 64, 64, 3])
        elegy.Model(r50, run_eagerly=True).predict(_x)
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
            sample_input=_x,
        )

        if weights == "coco":
            # models pretrained on coco have a fixed number of 73 output channels
            output_channels = 73
        super().__init__(
            output_channels,
            encoder=r50encoder,
            decoder=[512, 256, 128, 64, 32],
            dtype=dtype,
        )

        # manually injecting encoder into submodules, usually this happens after __init__
        # XXX: ugly ugly
        self._submodules.append("encoder")
        if weights is not None:
            load_pretrained_weights(self, weights, PRETRAINED_URLS)


_unet_rXX_init_docstring = """
Arguments:
    output_channels: Number of output classes. Ignored if `weights` is `'coco'`.
    weights: One of None (random initialization), `'coco'` (automatic download of
             weights pretrained on the COCO dataset) or a path to a weights file
    backbone_weights: Passed as the `weights=` arguments to the ResNet18 module.
                      Ignored if `weights` is not None.
    dtype: Optional dtype of the convolutions and linear operations,
           either jnp.float32 (default) or jnp.float16 for mixed precision.
"""

UNet_R18.__init__.__doc__ = _unet_rXX_init_docstring
UNet_R50.__init__.__doc__ = _unet_rXX_init_docstring


_unet_rXX_docstring = """

The pretrained models were trained on random image crops, resized to a fixed size of 384px and scaled between 0 and 1.
Only the following classes that are also present in the Pascal VOC dataset were used:
```
['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
 'train', 'boat', 'bird', 'cat', 'dog', 'horse', 'sheep',
 'cow', 'bottle', 'chair', 'couch', 'potted plant', 'dining table', 'tv']
```
This list is also accessible via `elegy.nets.unet.COCO_CLASSNAMES` and the corresponding class indices as `elegy.nets.unet.COCO_CHANNELS`.

***

Example inference:
```
fname,_  = urllib.request.urlretrieve('https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Take_ours%21.jpg/800px-Take_ours%21.jpg')
x        = np.array(PIL.Image.open(fname).resize([384,384])) / np.float32(255)
unet_r18 = elegy.nets.unet.UNet_R18(weights='coco')
y        = elegy.Model(unet_r18).predict(x[np.newaxis])[0]
i        = elegy.nets.unet.COCO_CLASSNAMES_TO_CHANNELS['sheep']

figure(0, (10,4))
subplot(121); imshow(x);               axis('off');
subplot(122); imshow(y.argmax(-1)==i); axis('off');
```
"""
UNet_R18.__doc__ += _unet_rXX_docstring
UNet_R50.__doc__ += _unet_rXX_docstring


# classes the pretrained unets were trained on (+background)
COCO_CLASSNAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "boat",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "bottle",
    "chair",
    "couch",
    "potted plant",
    "dining table",
    "tv",
]

# the corresponding channel numbers (+0 = background)
COCO_CHANNELS = [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]

COCO_CLASSNAMES_TO_CHANNELS = dict(zip(COCO_CLASSNAMES, COCO_CHANNELS))
COCO_CHANNELS_TO_CLASSNAMES = dict(zip(COCO_CHANNELS, COCO_CLASSNAMES))
