from elegy.types import DependencyUnavailable
from .generalized_module import GeneralizedModule
from . import elegy_module

try:
    from . import linen_module
except DependencyUnavailable as e:
    pass
