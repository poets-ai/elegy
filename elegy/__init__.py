__version__ = "0.1.4"

from haiku import get_parameter, get_state, set_state

from . import callbacks, losses, metrics, model, nn, regularizers
from .hooks import (
    add_loss,
    add_metric,
    add_summary,
    transform,
    TransformedState,
    HookStates,
    Context,
    context,
)
from .losses import Loss
from .metrics import Metric
from .model import Model
from .module import Module

from haiku import get_parameter, get_state, set_state
