import typing as tp
from typing import List, Tuple
import haiku as hk

from elegy import utils
from elegy.losses.loss import Loss


def get_unique_name(context, logs):
    context = list(context)
    context[0] += "_loss"

    name = "/".join(context)

    if name not in logs:
        return name

    i = 1
    while f"{name}_{i}" in logs:
        i += 1

    return f"{name}_{i}"


def forward_all(losses):
    def _losses_fn(**kwargs):

        logs = {}

        for context, loss_val in apply_recursive((), losses, **kwargs):
            loss_name = get_unique_name(context, logs)
            logs[loss_name] = loss_val

        return logs

    return _losses_fn


def apply_recursive(context: tp.Tuple[str, ...], losses, **kwargs):

    if isinstance(losses, tp.Callable):
        name = losses.name if isinstance(losses, Loss) else losses.__name__
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
