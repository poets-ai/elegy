from elegy import module
from typing import Tuple
from elegy.metrics.metric import Metric
import typing as tp


from elegy import utils
import jax


class Metrics(module.Module):
    def __init__(self, metrics):
        super().__init__(name="metrics")
        self.metrics = metrics

    def call(self, logs, **kwargs):

        # Loss logs
        count = module.get_state("count", initializer=0)
        total = module.get_state("total", initializer=jax.tree_map(lambda x: 0.0, logs))

        count += 1
        total = jax.tree_multimap(lambda a, b: a + b, total, logs)

        module.set_state("count", count)
        module.set_state("total", total)

        logs = jax.tree_map(lambda total: total / count, total)

        # Metric logs
        for context, val in apply_recursive((), self.metrics, **kwargs):
            name = "/".join(context)
            name = get_unique_name(logs, name)
            logs[name] = val

        return logs


def apply_recursive(context: tp.Tuple[str, ...], metrics, **kwargs):

    if isinstance(metrics, tp.Callable):

        name = (
            metrics.name
            if isinstance(metrics, module.Module)
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
