# isort:skip_file

__version__ = "0.8.6"

import elegy.types as types
import elegy.utils as utils

from treeo import Hashable, compact
from treex import Optimizer


from . import (
    callbacks,
    data,
    model,
    # nets,
    modules,
    strategies,
)

from .model import Trainer, Model
from .strategies import Strategy

# from .model.model_base import ModelBase, load
# from .model.model_core import (
#     GradStepOutput,
#     PredStepOutput,
#     TestStepOutput,
#     TrainStepOutput,
#     LossStepOutput,
#     ModelCore,
# )
from .types import KeySeq
from .modules.high_level.high_level_module import HighLevelModule
from .modules.managed.managed_module import ManagedModule
from .modules.module import CoreModule
from .modules.high_level.flax_module import FlaxModule
from .modules.managed.managed_flax_module import ManagedFlaxModule
from .pytree import PytreeObject, field, static_field
