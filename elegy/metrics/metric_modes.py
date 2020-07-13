import typing as tp
import haiku as hk

from elegy import utils
import jax


def match_outputs_and_labels(metrics_fn):
    def _metrics_fn(y_true, y_pred, **kwargs):

        if isinstance(metrics_fn, (tp.List, tp.Dict)):
            metrics = jax.tree_multimap(lambda f: f(), metrics_fn)
        else:
            metrics = [metrics_fn()]

        return {
            name: utils.inject_dependencies(metric)(
                y_true=y_true, y_pred=y_pred, **kwargs
            )
            for name, y_true, y_pred, metric in parse_structures(
                "", y_true, y_pred, metrics
            )
        }

    return _metrics_fn


def parse_structures(prefix, y_true, y_pred, metrics):

    if isinstance(y_true, (tp.Tuple, tp.List)):
        for i in range(len(y_true)):
            yield from parse_structures(
                f"{prefix}{i}/", y_true[i], y_pred[i], metrics[i]
            )
    elif isinstance(y_true, tp.Dict):
        for key in y_true:
            yield from parse_structures(
                f"{prefix}{key}/", y_true[key], y_pred[key], metrics[key]
            )
    elif isinstance(metrics, hk.Module):
        yield f"{prefix}{metrics.module_name}", y_true, y_pred, metrics

    elif isinstance(metrics, tp.Callable):
        yield f"{prefix}{metrics.__name__}", y_true, y_pred, metrics

    elif isinstance(metrics, tp.List):
        for metric in metrics:
            yield from parse_structures(prefix, y_true, y_pred, metric)

    else:
        raise ValueError(
            f"Invalid type for inputs or metrics, inputs {type(y_true)}, metrics {type(metrics)},"
        )


def get_mode_function(mode: str) -> tp.Callable:

    if mode == "match_outputs_and_labels":
        return match_outputs_and_labels
    elif mode == "manual":
        return lambda x: x
    else:
        raise ValueError(f"Mode '{mode}' not supported.")
