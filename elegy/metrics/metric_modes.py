from typing import Tuple
from elegy.metrics.metric import Metric
import typing as tp
import haiku as hk

from elegy import utils
import jax


def forward_all(metrics_fn):
    def _metrics_fn(**kwargs):

        if isinstance(metrics_fn, (tp.List, tp.Tuple, tp.Dict)):
            metrics = jax.tree_multimap(lambda f: f(), metrics_fn)
        else:
            metrics = metrics_fn()

        logs = {}

        for context, loss_val in apply_recursive((), metrics, **kwargs):
            name = "/".join(context)
            loss_name = get_unique_name(logs, name)
            logs[loss_name] = loss_val

        return logs

    return _metrics_fn


def apply_recursive(context: tp.Tuple[str, ...], losses, **kwargs):

    if isinstance(losses, tp.Callable):
        name = losses.module_name if isinstance(losses, Metric) else losses.__name__
        context += (name,)
        loss_val = utils.inject_dependencies(losses)(**kwargs)

        if isinstance(loss_val, tp.Dict):
            for name, loss_val in loss_val.items():
                yield context + (name,), loss_val
        else:
            yield context, loss_val

    elif isinstance(losses, (tp.Tuple, tp.List)):
        for loss in losses:
            yield from apply_recursive(context, loss, **kwargs)
    elif isinstance(losses, tp.Dict):
        for name, loss in losses.items():
            yield from apply_recursive(context + (name,), loss, **kwargs)
    else:
        raise TypeError(f"Invalid type {type(losses)}")


def get_unique_name(logs, name):

    if name not in logs:
        return name

    i = 1
    while f"{name}_{i}" in logs:
        i += 1

    return f"{name}_{i}"
