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
    update_context,
)
from elegy.types import Evaluation, OutputStates, Prediction, States, Training
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
from .model import Logs, Metrics, Model
from .model.model_base import ModelBase
from .module import Module, to_module
from .optimizer import Optimizer
from .types import DependencyUnavailable, Mode, RNGSeq, Uninitialized

try:
    from .model.generalized_module.linen_module import flax_summarize, flax_summary
except DependencyUnavailable as e:
    pass


__all__ = [
    "Loss",
    "Metric",
    "Model",
    "Module",
    "Optimizer",
    "RNG",
    "add_loss",
    "add_metric",
    "add_summary",
    "callbacks",
    "data",
    "get_dynamic_context",
    "get_losses",
    "get_metrics",
    "get_rng",
    "get_static_context",
    "get_summaries",
    "update_context",
    "initializers",
    "is_training",
    "jit",
    "losses",
    "metrics",
    "model",
    "module",
    "name_context",
    "nets",
    "next_key",
    "nn",
    "regularizers",
    "set_context",
    "set_rng",
    "set_training",
    "to_module",
    "training_context",
    "value_and_grad",
]
