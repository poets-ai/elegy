__version__ = "0.3.0"


from elegy.hooks import (
    add_loss,
    add_metric,
    add_summary,
    context,
    get_losses,
    get_metrics,
    get_summaries,
    jit,
    context,
)
from elegy import types
from elegy.utils import inject_dependencies

from . import losses  # module,
from . import (
    callbacks,
    data,
    hooks,
    initializers,
    metrics,
    model,
    nets,
    nn,
    regularizers,
    utils,
)
from .losses import Loss
from .metrics import Metric
from .model.model import Metrics, Model
from .model.model_core import PredStep, TestStep, GradStep, TrainStep
from .model.model_base import ModelBase
from .module import Module, to_module
from .optimizer import Optimizer
from .types import (
    DependencyUnavailable,
    RNGSeq,
    Uninitialized,
    States,
    OutputStates,
)
from .generalized_module.generalized_module import GeneralizedModule
from .generalized_optimizer.generalized_optimizer import GeneralizedOptimizer

try:
    from .generalized_module.linen_module import flax_summarize, flax_summary
except DependencyUnavailable as e:
    pass


__all__ = [
    "GeneralizedModule",
    "GeneralizedOptimizer",
    "GradStep",
    "Loss",
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
    "Uninitialized",
    "add_loss",
    "add_metric",
    "add_summary",
    "callbacks",
    "context",
    "context",
    "data",
    "flax_summarize",
    "flax_summary",
    "get_losses",
    "get_metrics",
    "get_summaries",
    "hooks",
    "initializers",
    "inject_dependencies",
    "jit",
    "losses",
    "metrics",
    "model",
    "nets",
    "nn",
    "regularizers",
    "to_module",
    "utils",
]
