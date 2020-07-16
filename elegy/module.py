from elegy.utils import Deferable
import typing as tp
import haiku as hk
from abc import abstractmethod


class Module(hk.Module, Deferable):
    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    @abstractmethod
    def call(self, *args, **kwargs):
        ...
