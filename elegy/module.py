import typing as tp
from abc import abstractmethod

import haiku as hk
import jax.numpy as jnp
import numpy as np

from elegy import utils
from elegy.utils import Deferable


class Module(hk.Module, Deferable):
    """Basic Elegy Module
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.call = utils.inject_dependencies(self.call)

    def __call__(self, *args, **kwargs):

        outputs = self.call(*args, **kwargs)

        if utils.LOCAL.calculating_summary:
            layer_tag = f"__ELEGY__LAYEROUTPUT__{utils.LOCAL.layer_count}"
            hk.get_state(layer_tag, [], init=lambda *args: np.array(0.0))
            utils.LOCAL.layer_count += 1
            hk.set_state(layer_tag, outputs)

        return outputs

    @abstractmethod
    def call(self, *args, **kwargs):
        ...
