import typing as tp
from typing import List, Tuple

from elegy import utils
from elegy.losses.loss import Loss
from elegy.module import Module


class Losses(Module):
    def __init__(self, losses):
        super().__init__(name="losses")
        self.losses = losses

    def call(self, **kwargs):

        logs = {}

        for context, val in apply_recursive((), self.losses, **kwargs):
            loss_name = get_unique_name(context, logs)
            logs[loss_name] = val

        return logs


def apply_recursive(context: tp.Tuple[str, ...], losses, **kwargs):

    if isinstance(losses, tp.Callable):
        name = (
            losses.name
            if isinstance(losses, Loss)
            else utils.lower_snake_case(losses.__name__)
        )
        context += (name,)
        val = utils.inject_dependencies(losses)(**kwargs)

        if isinstance(val, tp.Dict):
            for name, val in val.items():
                yield context + (name,), val
        else:
            yield context, val

    elif isinstance(losses, (tp.Tuple, tp.List)):
        for loss in losses:
            yield from apply_recursive(context, loss, **kwargs)
    elif isinstance(losses, tp.Dict):
        for name, loss in losses.items():
            yield from apply_recursive(context + (name,), loss, **kwargs)
    else:
        raise TypeError(f"Invalid type {type(losses)}")


def get_unique_name(context, logs):
    context = list(context)

    if not context[0].endswith("loss"):
        context[0] += "_loss"

    name = "/".join(context)

    if name not in logs:
        return name

    i = 1
    while f"{name}_{i}" in logs:
        i += 1

    return f"{name}_{i}"
