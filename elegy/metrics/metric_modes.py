from typing import Tuple
from elegy.metrics.metric import Metric
import typing as tp
import haiku as hk

from elegy import utils
import jax


def forward_all(metrics):
    def _metrics_fn(**kwargs):

        logs = {}

        for context, val in apply_recursive((), metrics, **kwargs):
            name = "/".join(context)
            loss_name = get_unique_name(logs, name)
            logs[loss_name] = val

        return logs

    return _metrics_fn


def apply_recursive(context: tp.Tuple[str, ...], metrics, **kwargs):

    if isinstance(metrics, tp.Callable):

        name = (
            metrics.module_name if isinstance(metrics, hk.Module) else metrics.__name__
        )
        context += (name,)
        loss_val = utils.inject_dependencies(metrics)(**kwargs)

        if isinstance(loss_val, tp.Dict):
            for name, loss_val in loss_val.items():
                yield context + (name,), loss_val
        else:
            yield context, loss_val

    elif isinstance(metrics, (tp.Tuple, tp.List)):
        for loss in metrics:
            yield from apply_recursive(context, loss, **kwargs)
    elif isinstance(metrics, tp.Dict):
        for name, loss in metrics.items():
            yield from apply_recursive(context + (name,), loss, **kwargs)
    else:
        raise TypeError(f"Invalid type {type(metrics)}")


def get_unique_name(logs, name):

    if name not in logs:
        return name

    i = 1
    while f"{name}_{i}" in logs:
        i += 1

    return f"{name}_{i}"
