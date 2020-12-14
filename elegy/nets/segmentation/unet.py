import elegy
from elegy import module
import jax, jax.numpy as jnp


#TODO: docs
#TODO: typing
#TODO: mkdocs

'''
**TODO**
- depth
- number of convs in convblock
- backbone
- dilation
- bn after relu + no BN scaling
- separable convs
'''




class ConvBlock(module.Module):
    def __init__(self, channels, batchnorm=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channels = channels
        self.batchnorm = batchnorm
        
    def call(self, x):
        x = elegy.nn.Conv2D(self.channels, (3,3), padding='SAME', with_bias=not self.batchnorm)(x)
        x = elegy.nn.BatchNormalization(decay_rate=0.9)(x) if self.batchnorm else x
        x = elegy.to_module(jax.nn.relu)()(x)
        x = elegy.nn.Conv2D(self.channels, (3,3), padding='SAME', with_bias=not self.batchnorm)(x)
        x = elegy.nn.BatchNormalization(decay_rate=0.9)(x) if self.batchnorm else x
        x = elegy.to_module(jax.nn.relu)()(x)
        return x

class DownBlock(module.Module):
    def __init__(self, batchnorm=True, *args, **kwargs):
        super().__init__()
        self.batchnorm=batchnorm
    def call(self, x):
        x = elegy.nn.MaxPool(window_shape=(1,2,2,1), strides=(1,2,2,1), padding='SAME')(x)
        x = ConvBlock(channels=x.shape[-1]*2, batchnorm=self.batchnorm)(x)
        return x

class UpBlock(module.Module):
    def __init__(self, batchnorm=True, *args, **kwargs):
        super().__init__()
        self.batchnorm=batchnorm
    def call(self, x, skipx):
        x = jax.image.resize(x, skipx.shape, method='bilinear')
        x = elegy.nn.Conv2D(skipx.shape[-1], (1,1))(x)
        x = jnp.concatenate([x,skipx], axis=-1)
        x = ConvBlock(channels=skipx.shape[-1], batchnorm=self.batchnorm)(x)
        return x

class UNet(module.Module):
    def __init__(self, output_channels, batchnorm=True):
        super().__init__()
        self.output_channels = output_channels
        self.batchnorm = batchnorm
    
    def call(self, x):
        x = x0 = ConvBlock(channels=64, batchnorm=self.batchnorm)(x)
        x = x1 = DownBlock()(x)
        x = x2 = DownBlock()(x)
        x = x3 = DownBlock()(x)
        x      = DownBlock()(x)
        x      = UpBlock()(x,x3)
        x      = UpBlock()(x,x2)
        x      = UpBlock()(x,x1)
        x      = UpBlock()(x,x0)
        x      = elegy.nn.Conv2D(self.output_channels, (1,1))(x)
        return x
