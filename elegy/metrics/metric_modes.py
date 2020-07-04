import typing as tp
import haiku as hk
from elegy.utils.dependency_injection import DIFunction


def get_mode_function(mode: str) -> tp.Callable:

    if mode == "match_outputs_and_labels":
        return match_outputs_and_labels
    elif mode == "forward_all":
        return forward_all
    elif mode == "manual":
        return lambda x: x
    else:
        raise ValueError(f"Mode '{mode}' not supported.")


def forward_all(
    modules_fn: tp.Callable[
        [],
        tp.Union[
            hk.Module,
            tp.List[tp.Union[hk.Module, tp.Tuple[str, tp.Callable]]],
            tp.Dict[str, tp.Callable],
        ],
    ]
):
    def _metrics_fn(y_true, y_pred, **kwargs):

        metrics = modules_fn()

        if isinstance(metrics, hk.Module):
            metrics = [metrics]

        if isinstance(metrics, tp.Dict):
            metrics = metrics.items()
        else:
            metrics = (
                (metric.module_name, metric)
                if isinstance(metric, hk.Module)
                else metric
                for metric in metrics
            )

        return {
            name: DIFunction.create(metric)(y_true, y_pred, **kwargs)
            for name, metric in metrics
        }

    return _metrics_fn


KeyValueLike = tp.Union[hk.Module, tp.Tuple[str, tp.Callable]]
DictLike = tp.Union[tp.Dict[str, tp.Callable], tp.List[KeyValueLike]]


def match_outputs_and_labels(
    modules_fn: tp.Callable[
        [],
        tp.Union[
            KeyValueLike,
            tp.List[tp.Union[KeyValueLike, DictLike]],
            tp.Dict[str, tp.Union[KeyValueLike, DictLike]],
        ],
    ]
):
    def _metrics_fn(y_true, y_pred, **kwargs):

        metrics = modules_fn()

        return {
            name: DIFunction.create(metric)(y_true, y_pred, **kwargs)
            for name, y_true, y_pred, metric in parse_structures(
                "", y_true, y_pred, metrics
            )
        }

    return _metrics_fn


def parse_structures(prefix, y_true, y_pred, metrics):

    if isinstance(y_true, tp.Tuple):
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

    elif isinstance(metrics, tp.Tuple):
        yield f"{prefix}{metrics[0]}", y_true, y_pred, metrics[1]

    elif isinstance(metrics, tp.List):
        for metric in metrics:
            yield from parse_structures(prefix, y_true, y_pred, metric)

    else:
        raise ValueError(
            f"Invalid type for inputs or metrics, inputs {type(y_true)}, metrics {type(metrics)},"
        )

