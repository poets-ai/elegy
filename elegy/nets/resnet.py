# adapted from the flax library https://github.com/google/flax

import jax, jax.numpy as jnp
from elegy import module, nn
import typing as tp


__all__ = [
    "ResNet",
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "ResNet152",
    "ResNet200",
]


class ResNetBlock(module.Module):
    """ResNet (identity) block"""

    def __init__(
        self,
        n_filters: int,
        strides: tp.Optional[tp.Tuple[int]] = (1, 1),
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.n_filters = n_filters
        self.strides = strides

    def call(self, x: jnp.ndarray):
        x0 = x
        x = nn.Conv2D(
            self.n_filters,
            (3, 3),
            with_bias=False,
            stride=self.strides,
            dtype=self.dtype,
        )(x)
        x = nn.BatchNormalization(decay_rate=0.9, eps=1e-5)(x)
        x = jax.nn.relu(x)

        x = nn.Conv2D(self.n_filters, (3, 3), with_bias=False, dtype=self.dtype)(x)
        x = nn.BatchNormalization(decay_rate=0.9, eps=1e-5)(x)

        if x0.shape != x.shape:
            x0 = nn.Conv2D(
                self.n_filters,
                (1, 1),
                with_bias=False,
                stride=self.strides,
                dtype=self.dtype,
            )(x0)
            x0 = nn.BatchNormalization(decay_rate=0.9, eps=1e-5)(x0)
        return jax.nn.relu(x0 + x)


class BottleneckResNetBlock(ResNetBlock):
    """ResNet Bottleneck block."""

    def call(self, x: jnp.ndarray):
        x0 = x
        x = nn.Conv2D(self.n_filters, (1, 1), with_bias=False, dtype=self.dtype)(x)
        x = nn.BatchNormalization(decay_rate=0.9, eps=1e-5)(x)
        x = jax.nn.relu(x)
        x = nn.Conv2D(
            self.n_filters,
            (3, 3),
            with_bias=False,
            stride=self.strides,
            dtype=self.dtype,
        )(x)
        x = nn.BatchNormalization(decay_rate=0.9, eps=1e-5)(x)
        x = jax.nn.relu(x)
        x = nn.Conv2D(self.n_filters * 4, (1, 1), with_bias=False, dtype=self.dtype)(x)
        x = nn.BatchNormalization(decay_rate=0.9, eps=1e-5, scale_init=jnp.zeros)(x)

        if x0.shape != x.shape:
            x0 = nn.Conv2D(
                self.n_filters * 4,
                (1, 1),
                with_bias=False,
                stride=self.strides,
                dtype=self.dtype,
            )(x0)
            x0 = nn.BatchNormalization(decay_rate=0.9, eps=1e-5)(x0)
        return jax.nn.relu(x0 + x)


class ResNet(module.Module):
    """A generic ResNet V1 architecture that can be customized for non-standard configurations
    Original Paper: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
    """

    __all__ = ["__init__", "call"]

    def __init__(
        self,
        stages: tp.List[int],
        block_type: tp.Union[ResNetBlock, BottleneckResNetBlock],
        lowres: tp.Optional[bool] = False,
        *args,
        **kwargs
    ):
        """
        Arguments:
            stages: A list of integers representing the number of blocks in each stage.
                    e.g: [3, 4, 6, 3] for a ResNet50
            block_type: Which ResNet block type to use.
            lowres: Optional, whether to use the low resolution version
                    as described in subsection 4.2 of the orignal paper.
                    This version is better suited for datasets like CIFAR10. (Default: False)
            dtype: Optional dtype of the convolutions and linear operations,
                    either jnp.float32 (default) or jnp.float16 for mixed precision.
        """

        super().__init__(*args, **kwargs)
        self.stages = stages
        self.block_type = block_type
        self.lowres = lowres

    def call(self, x: jnp.ndarray):
        x = nn.Conv2D(
            64,
            (7, 7) if not self.lowres else (3, 3),
            stride=(2, 2) if not self.lowres else (1, 1),
            padding="SAME",
            with_bias=False,
            dtype=self.dtype,
        )(x)
        x = nn.BatchNormalization(decay_rate=0.9, eps=1e-5)(x)
        x = module.to_module(jax.nn.relu)()(x)

        if not self.lowres:
            x = nn.MaxPool(
                window_shape=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding="SAME"
            )(x)
        for i, block_size in enumerate(self.stages):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_type(64 * 2 ** i, strides=strides, dtype=self.dtype)(x)
        GAP = lambda x: jnp.mean(x, axis=(1, 2))
        x = module.to_module(GAP)(name="global_average_pooling")(x)
        x = nn.Linear(1000, dtype=self.dtype)(x)
        to_float32 = lambda x: jnp.asarray(x, jnp.float32)
        x = module.to_module(to_float32)(name="to_float32")(x)
        return x


class ResNet18(ResNet):
    def __init__(
        self,
        lowres: tp.Optional[bool] = False,
        dtype: tp.Optional[tp.Union["float16", "float32"]] = "float32",
        *args,
        **kwargs
    ):
        super().__init__(
            stages=[2, 2, 2, 2],
            block_type=ResNetBlock,
            lowres=lowres,
            dtype=dtype,
            *args,
            **kwargs
        )


class ResNet34(ResNet):
    def __init__(
        self,
        lowres: tp.Optional[bool] = False,
        dtype: tp.Optional[tp.Union["float16", "float32"]] = "float32",
        *args,
        **kwargs
    ):
        super().__init__(
            stages=[3, 4, 6, 3],
            block_type=ResNetBlock,
            lowres=lowres,
            dtype=dtype,
            *args,
            **kwargs
        )


class ResNet50(ResNet):
    def __init__(
        self,
        lowres: tp.Optional[bool] = False,
        dtype: tp.Optional[tp.Union["float16", "float32"]] = "float32",
        *args,
        **kwargs
    ):
        super().__init__(
            stages=[3, 4, 6, 3],
            block_type=BottleneckResNetBlock,
            lowres=lowres,
            dtype=dtype,
            *args,
            **kwargs
        )


class ResNet101(ResNet):
    def __init__(
        self,
        lowres: tp.Optional[bool] = False,
        dtype: tp.Optional[tp.Union["float16", "float32"]] = "float32",
        *args,
        **kwargs
    ):
        super().__init__(
            stages=[3, 4, 23, 3],
            block_type=BottleneckResNetBlock,
            lowres=lowres,
            dtype=dtype,
            *args,
            **kwargs
        )


class ResNet152(ResNet):
    def __init__(
        self,
        lowres: tp.Optional[bool] = False,
        dtype: tp.Optional[tp.Union["float16", "float32"]] = "float32",
        *args,
        **kwargs
    ):
        super().__init__(
            stages=[3, 8, 36, 3],
            block_type=BottleneckResNetBlock,
            lowres=lowres,
            dtype=dtype,
            *args,
            **kwargs
        )


class ResNet200(ResNet):
    def __init__(
        self,
        lowres: tp.Optional[bool] = False,
        dtype: tp.Optional[tp.Union["float16", "float32"]] = "float32",
        *args,
        **kwargs
    ):
        super().__init__(
            stages=[3, 24, 36, 3],
            block_type=BottleneckResNetBlock,
            lowres=lowres,
            dtype=dtype,
            *args,
            **kwargs
        )


_resnet__init___docstring = """
Instantiates the {} architecture from [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

Arguments:
    lowres: Optional, whether to use the low resolution version
            as described in subsection 4.2 of the orignal paper.
            This version is better suited for datasets like CIFAR10. (Default: False)
    dtype: Optional dtype of the convolutions and linear operations, 
           either jnp.float32 (default) or jnp.float16 for mixed precision.
"""

ResNet18.__init__.__doc__ = _resnet__init___docstring.format("ResNet18")
ResNet34.__init__.__doc__ = _resnet__init___docstring.format("ResNet34")
ResNet50.__init__.__doc__ = _resnet__init___docstring.format("ResNet50")
ResNet101.__init__.__doc__ = _resnet__init___docstring.format("ResNet101")
ResNet152.__init__.__doc__ = _resnet__init___docstring.format("ResNet152")
ResNet200.__init__.__doc__ = _resnet__init___docstring.format("ResNet200")
