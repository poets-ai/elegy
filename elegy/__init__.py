__version__ = "0.7.1"


from elegy import types
from elegy.utils import inject_dependencies

from . import losses  # module,
from . import (
    callbacks,
    hooks,
    initializers,
    metrics,
    model,
    nets,
    nn,
    regularizers,
    utils,
    data,
)
from .losses import Loss
from .metrics import Metric
from .model.model import Metrics, Model, Losses
from .model.model_core import PredStep, TestStep, GradStep, TrainStep
from .model.model_base import ModelBase, load
from .module import Module, to_module
from .optimizer import Optimizer
from .types import (
    RNGSeq,
    States,
    OutputStates,
)
from .generalized_module.generalized_module import GeneralizedModule
from .generalized_optimizer.generalized_optimizer import GeneralizedOptimizer

try:
    from .generalized_module.linen_module import flax_summarize, flax_summary
except types.DependencyUnavailable as e:
    pass
try:
    from .generalized_module.haiku_module import (
        HaikuModule,
        haiku_summarize,
        haiku_summary,
    )
except types.DependencyUnavailable as e:
    pass


__all__ = [
    "GeneralizedModule",
    "GeneralizedOptimizer",
    "GradStep",
    "HaikuModule",
    "Loss",
    "Losses",
    "Metric",
    "Metrics",
    "Model",
    "ModelBase",
    "Module",
    "Optimizer",
    "OutputStates",
    "PredStep",
    "RNGSeq",
    "States",
    "TestStep",
    "TrainStep",
    "callbacks",
    "data",
    "flax_summarize",
    "flax_summary",
    "hooks",
    "initializers",
    "inject_dependencies",
    "losses",
    "metrics",
    "nets",
    "nn",
    "regularizers",
    "to_module",
]
