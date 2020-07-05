import typing as tp
import haiku as hk
import jax.numpy as jnp
from abc import abstractmethod
from elegy import utils


class Metric(hk.Module):
    def __init__(
        self, name: tp.Optional[str] = None, dtype: tp.Optional[jnp.dtype] = None
    ):
        super().__init__(name=name)

        self._dtype = self._dtype = dtype if dtype is not None else jnp.float32

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    @abstractmethod
    def call(self, *args, **kwargs):
        ...

