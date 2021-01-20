__version__ = "0.3.0"


from elegy.hooks import (
    add_loss,
    add_metric,
    add_summary,
    get_losses,
    get_metrics,
    get_rng,
    get_summaries,
    get_training,
    hooks_context,
    is_training,
    jit,
    next_key,
    value_and_grad,
)
from elegy.types import Evaluation, OutputStates, Prediction, States, Training

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
from .model import Logs, Metrics, Model, Optimizer
from .model.model_base import ModelBase
from .module import Module, get_module_path, get_module_path_str
from .types import Mode, RNGSeq, Uninitialized

# from .module import (
#     RNG,
#     LocalContext,
#     Module,
#     add_loss,
#     add_metric,
#     add_summary,
#     get_dynamic_context,
#     get_losses,
#     get_metrics,
#     get_rng,
#     get_static_context,
#     get_summaries,
#     hooks_context,
#     is_training,
#     jit,
#     name_context,
#     next_key,
#     set_context,
#     set_rng,
#     set_training,
#     to_module,
#     training_context,
#     value_and_grad,
# )

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
    "hooks_context",
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
