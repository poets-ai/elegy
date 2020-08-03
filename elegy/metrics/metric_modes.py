from elegy.module import Module
from typing import Tuple
from elegy.metrics.metric import Metric
import typing as tp


from elegy import utils
import jax


class Metrics(Module):
    def __init__(self, metrics):
        self.metrics = metrics

    def call(self, **kwargs):

        logs = {}

        for context, val in apply_recursive((), self.metrics, **kwargs):
            name = "/".join(context)
            name = get_unique_name(logs, name)
            logs[name] = val

        return logs


def apply_recursive(context: tp.Tuple[str, ...], metrics, **kwargs):

    if isinstance(metrics, tp.Callable):

        name = (
            metrics.name
            if isinstance(metrics, Module)
            else utils.lower_snake_case(metrics.__name__)
        )
        context += (name,)
        value = utils.inject_dependencies(metrics)(**kwargs)

        if isinstance(value, tp.Dict):
            for name, value in value.items():
                yield context + (name,), value
        else:
            yield context, value

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
