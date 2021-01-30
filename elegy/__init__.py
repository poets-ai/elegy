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
from .model import Metrics, Model
from .model.model_core import PredStep, TestStep, GradStep, TrainStep
from .model.model_base import ModelBase
from .module import Module, to_module
from .optimizer import Optimizer
from .types import DependencyUnavailable, Mode, RNGSeq, Uninitialized, States

try:
    from .model.generalized_module.linen_module import flax_summarize, flax_summary
except DependencyUnavailable as e:
    pass


__all__ = [
    "add_loss",
    "add_metric",
    "add_summary",
    "context",
    "get_losses",
    "get_metrics",
    "get_summaries",
    "jit",
    "context",
    "States",
    "inject_dependencies",
    "losses",
    "callbacks",
    "data",
    "hooks",
    "initializers",
    "metrics",
    "model",
    "nets",
    "nn",
    "regularizers",
    "utils",
    "Loss",
    "Metric",
    "Metrics",
    "Model",
    "PredStep",
    "TestStep",
    "GradStep",
    "TrainStep",
    "ModelBase",
    "Module",
    "to_module",
    "Optimizer",
    "DependencyUnavailable",
    "Mode",
    "RNGSeq",
    "Uninitialized",
    "flax_summarize",
    "flax_summary",
]
