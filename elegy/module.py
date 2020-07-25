import typing as tp
from abc import abstractmethod

import haiku as hk
import jax.numpy as jnp
import numpy as np

from elegy import utils
from elegy.utils import Deferable
from elegy import hooks


class Module(hk.Module, Deferable):
    """Basic Elegy Module
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.call = utils.inject_dependencies(self.call)

    def __call__(self, *args, **kwargs):

        outputs = self.call(*args, **kwargs)

        hooks.add_summary("outputs", outputs)

        return outputs

    @abstractmethod
    def call(self, *args, **kwargs):
        ...
