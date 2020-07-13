import typing as tp
import haiku as hk

from elegy import utils
from elegy.losses.loss import Loss

KeyValueLike = tp.Union[hk.Module, tp.Tuple[str, tp.Callable]]
DictLike = tp.Union[tp.Dict[str, tp.Callable], tp.List[KeyValueLike]]


def match_outputs_and_labels(loss_fns):
    def _losses_fn(y_true, y_pred, **kwargs):

        logs = {}

        for name, y_true, y_pred, loss in parse_structures(
            "", y_true, y_pred, loss_fns
        ):
            loss_name = get_unique_name(logs, name)

            logs[loss_name] = utils.inject_dependencies(loss)(
                y_true=y_true, y_pred=y_pred, **kwargs
            )

        return logs

    return _losses_fn


def get_unique_name(losses, name):
    loss_name = f"{name}_loss"

    if loss_name not in losses:
        return loss_name

    i = 1
    while f"{loss_name}_{i}" in losses:
        i += 1

    return f"{loss_name}_{i}"


def parse_structures(prefix, y_true, y_pred, losses):

    if isinstance(y_true, (tp.Tuple, tp.List)):
        for i in range(len(y_true)):
            yield from parse_structures(
                f"{prefix}{i}/", y_true[i], y_pred[i], losses[i]
            )
    elif isinstance(y_true, tp.Dict):
        for key in y_true:
            yield from parse_structures(
                f"{prefix}{key}/", y_true[key], y_pred[key], losses[key]
            )
    elif isinstance(losses, Loss):
        yield f"{prefix}{losses.name}", y_true, y_pred, losses

    elif isinstance(losses, tp.Tuple):
        yield f"{prefix}{losses[0]}", y_true, y_pred, losses[1]

    elif isinstance(losses, tp.List):
        for metric in losses:
            yield from parse_structures(prefix, y_true, y_pred, metric)

    else:
        raise ValueError(
            f"Invalid type for inputs or losses, inputs {type(y_true)}, losses {type(losses)},"
        )


def get_mode_function(mode: str) -> tp.Callable:

    if mode == "match_outputs_and_labels":
        return match_outputs_and_labels
    elif mode == "manual":
        return lambda x: x
    else:
        raise ValueError(f"Mode '{mode}' not supported.")


def get_aux_losses_fn(loss_fns_):
    def _aux_losses(*args, **kwargs):

        loss_fns = loss_fns_
        aux_losses = {}

        if not isinstance(loss_fns, (tp.List, tp.Tuple)):
            loss_fns = [loss_fns]

        # TODO: refactor this into a single function
        for aux_loss in loss_fns:

            if isinstance(aux_loss, tp.Tuple):
                name, aux_loss = aux_loss
            else:
                name = aux_loss.name

            name = get_unique_name(aux_losses, name)

            aux_losses[name] = aux_loss(*args, **kwargs)

        return aux_losses

    return _aux_losses
