#adapted from the flax library https://github.com/google/flax

import jax, jax.numpy as jnp
from elegy import module,nn


class ResNetBlock(module.Module):
    '''ResNet (identity) block'''
    def call(self, x, n_filters, strides=(1, 1)):
        x0 = x
        x  = nn.Conv2D(n_filters, (3,3), with_bias=False, stride=strides, dtype=self.dtype)(x)
        x  = nn.BatchNormalization(decay_rate=0.9, eps=1e-5)(x)
        x  = jax.nn.relu(x)
        
        x  = nn.Conv2D(n_filters, (3,3), with_bias=False, dtype=self.dtype)(x)
        x  = nn.BatchNormalization(decay_rate=0.9, eps=1e-5)(x)
        
        if x0.shape != x.shape:
            x0 = nn.Conv2D(n_filters, (1,1), with_bias=False, stride=strides, dtype=self.dtype)(x0)
            x0 = nn.BatchNormalization(decay_rate=0.9, eps=1e-5)(x0)
        return jax.nn.relu(x0 + x)


class BottleneckResNetBlock(module.Module):
    '''ResNet Bottleneck block.'''
    def call(self, x, n_filters, strides=(1, 1)):
        x0 = x
        x  = nn.Conv2D(n_filters, (1,1), with_bias=False, dtype=self.dtype)(x)
        x  = nn.BatchNormalization(decay_rate=0.9, eps=1e-5)(x)
        x  = jax.nn.relu(x)
        x  = nn.Conv2D(n_filters, (3,3), with_bias=False, stride=strides, dtype=self.dtype)(x)
        x  = nn.BatchNormalization(decay_rate=0.9, eps=1e-5)(x)
        x  = jax.nn.relu(x)
        x  = nn.Conv2D(n_filters*4, (1,1), with_bias=False, dtype=self.dtype)(x)
        x  = nn.BatchNormalization(decay_rate=0.9, eps=1e-5, scale_init=jnp.zeros)(x)

        if x0.shape != x.shape:
            x0 = nn.Conv2D(n_filters*4, (1,1), with_bias=False, stride=strides, dtype=self.dtype)(x0)
            x0 = nn.BatchNormalization(decay_rate=0.9, eps=1e-5)(x0)
        return jax.nn.relu(x0 + x)



class ResNet(module.Module):
    '''ResNet V1'''
    def __init__(self, stages, block_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stages     = stages
        self.block_type = block_type
    
    def call(self, x):
        x       = nn.Conv2D(64, (7,7), stride=(2,2), padding='SAME', with_bias=False, dtype=self.dtype)(x)
        x       = nn.BatchNormalization(decay_rate=0.9, eps=1e-5)(x)
        x       = jax.nn.relu(x)
        
        x       = nn.linear.hk.max_pool(x, window_shape=(1,3,3,1), strides=(1,2,2,1), padding="SAME")
        for i, block_size in enumerate(self.stages):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x       = self.block_type(dtype=self.dtype)(x, 64*2**i, strides=strides)
        x       = jnp.mean(x, axis=(1, 2))
        x       = nn.Linear(1000, dtype=self.dtype)(x)
        x       = jnp.asarray(x, jnp.float32)
        return x


class ResNet18(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(stages=[2,2,2,2], block_type=ResNetBlock, *args, **kwargs)

class ResNet34(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(stages=[3,4,6,3], block_type=ResNetBlock, *args, **kwargs)

class ResNet50(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(stages=[3,4,6,3], block_type=BottleneckResNetBlock, *args, **kwargs)

class ResNet101(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(stages=[3,4,23,3], block_type=BottleneckResNetBlock, *args, **kwargs)

class ResNet152(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(stages=[3,8,36,3], block_type=BottleneckResNetBlock, *args, **kwargs)

class ResNet200(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(stages=[3,24,36,3], block_type=BottleneckResNetBlock, *args, **kwargs)
