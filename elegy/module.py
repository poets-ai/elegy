import typing as tp
from abc import abstractmethod

import haiku as hk

from elegy import utils
from elegy.utils import Deferable


class Module(hk.Module, Deferable):
    """Basic Elegy Module
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.call = utils.inject_dependencies(self.call)

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    @abstractmethod
    def call(self, *args, **kwargs):
        ...
