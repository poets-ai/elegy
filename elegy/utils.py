from elegy.types import Index, Path, States
import functools
import inspect
import re
import sys
import typing as tp
from functools import total_ordering

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np
import toolz
from deepmerge import always_merger

from elegy.frozen_dict import FrozenDict


def maybe_expand_dims(a: np.ndarray, b: np.ndarray) -> tp.Tuple[np.ndarray, np.ndarray]:
    assert np.prod(a.shape) == np.prod(b.shape)

    if a.ndim < b.ndim:
        a = a[..., None]

    if b.ndim < a.ndim:
        b = b[..., None]

    return a, b


def wraps(f, docs: bool = True):
    assigments = ("__annotations__",)

    if docs:
        assigments += ("__doc__",)

    return functools.wraps(f, assigned=assigments, updated=())


def inject_dependencies(
    f: tp.Callable,
    signature_f: tp.Optional[tp.Callable] = None,
    rename: tp.Optional[tp.Dict[str, str]] = None,
):
    if signature_f is None:
        signature_f = f

    f_params = get_function_args(signature_f)

    @functools.wraps(signature_f)
    def wrapper(*args, **kwargs):
        n_args = len(args)
        arg_names = [arg.name for arg in f_params[:n_args]]
        kwarg_names = [arg.name for arg in f_params[n_args:]]

        if rename:
            for old, new in rename.items():
                if old in kwargs:
                    kwargs[new] = kwargs.pop(old)

        if not any(arg.kind == inspect.Parameter.VAR_KEYWORD for arg in f_params):
            # print(list(kwargs.keys()))
            # print(kwarg_names)
            kwargs = {
                arg: kwargs[arg]
                for arg in kwarg_names
                if arg not in arg_names and arg in kwargs
            }

        return f(*args, **kwargs)

    return wrapper


def get_function_args(f) -> tp.List[inspect.Parameter]:
    return list(inspect.signature(f).parameters.values())


def get_input_args(
    x: tp.Union[np.ndarray, jnp.ndarray, tp.Dict[str, tp.Any], tp.Tuple],
    *,
    states: States,
    training: bool,
) -> tp.Tuple[tp.Tuple, tp.Dict[str, tp.Any]]:

    if isinstance(x, tp.Tuple):
        args = x
        kwargs = {}
    elif isinstance(x, tp.Dict):
        args = ()
        kwargs = x
    else:
        args = (x,)
        kwargs = {}

    apply_kwargs = dict(
        training=training,
        net_params=states.net_params,
        net_states=states.net_states,
        rng=states.rng,
        states=states,
    )
    apply_kwargs.update(kwargs)

    return args, apply_kwargs


def split(
    d: tp.Union[tp.Dict[str, tp.Any], tp.Mapping[str, tp.Any]]
) -> tp.Iterable[tp.Dict[str, tp.Any]]:

    for k, v in d.items():

        parts = k.split("/")
        parts.reverse()

        if isinstance(v, (tp.Dict, tp.Mapping)):
            vs = list(split(v))
        else:
            vs = [v]

        for v in vs:
            output = {}

            for k in parts:
                if not output:
                    output[k] = v
                else:
                    output = {k: output}

            yield output


def split_and_merge(
    d: tp.Union[tp.Dict[str, tp.Any], tp.Mapping[str, tp.Any]]
) -> tp.Dict[str, tp.Any]:

    ds = split(d)
    return toolz.reduce(always_merger.merge, ds, {})


def lower_snake_case(s: str) -> str:
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()
    parts = s.split("_")
    output_parts = []

    for i in range(len(parts)):
        if i == 0 or len(parts[i - 1]) > 1:
            output_parts.append(parts[i])
        else:
            output_parts[-1] += parts[i]

    return "_".join(output_parts)


def get_name(obj) -> str:
    if hasattr(obj, "name") and obj.name:
        return obj.name
    elif hasattr(obj, "__name__") and obj.__name__:
        return obj.__name__
    elif hasattr(obj, "__class__") and obj.__class__.__name__:
        return lower_snake_case(obj.__class__.__name__)
    else:
        raise ValueError(f"Could not get name for: {obj}")


def to_static(structure: tp.Any) -> tp.Any:

    if isinstance(structure, (tp.Dict, FrozenDict)):
        return tuple((k, to_static(v)) for k, v in structure.items())
    elif isinstance(structure, (tp.List, tp.Tuple)):
        return tuple(to_static(v) for v in structure)
    else:
        return structure


def _leaf_paths(path: Path, inputs: tp.Any) -> tp.Iterable[tp.Tuple[Path, tp.Any]]:

    if isinstance(inputs, (tp.Tuple, tp.List)):
        for i, value in enumerate(inputs):
            yield from _leaf_paths(path + (i,), value)
    elif isinstance(inputs, tp.Dict):
        for name, value in inputs.items():
            yield from _leaf_paths(path + (name,), value)
    else:
        yield (path, inputs)


def leaf_paths(inputs: tp.Any) -> tp.List[tp.Tuple[Path, tp.Any]]:
    return list(_leaf_paths((), inputs))


def _flatten_names(path: Path, inputs: tp.Any) -> tp.Iterable[tp.Tuple[Path, tp.Any]]:

    if isinstance(inputs, (tp.Tuple, tp.List)):
        for i, value in enumerate(inputs):
            yield from _flatten_names(path, value)
    elif isinstance(inputs, tp.Dict):
        for name, value in inputs.items():
            yield from _flatten_names(path + (name,), value)
    else:
        yield (path, inputs)


def flatten_names(inputs: tp.Any) -> tp.List[tp.Tuple[str, tp.Any]]:
    return [
        ("/".join(map(str, path)), value) for path, value in _flatten_names((), inputs)
    ]


def get_unique_name(
    names: tp.Set[str],
    name: str,
):

    if name in names:
        i = 1
        while f"{name}_{i}" in names:
            i += 1

        name = f"{name}_{i}"

    names.add(name)
    return name


def merge_with_unique_names(
    a: tp.Dict[str, tp.Any],
    *rest: tp.Dict[str, tp.Any],
) -> tp.Dict[str, tp.Any]:

    a = a.copy()

    for b in rest:
        a = _merge_with_unique_names(a, b)

    return a


def _merge_with_unique_names(
    a: tp.Dict[str, tp.Any],
    b: tp.Dict[str, tp.Any],
) -> tp.Dict[str, tp.Any]:
    names = set()
    output = dict(a)

    for name, value in b.items():
        output[get_unique_name(names, name)] = value

    return output


def parameters_count(params: tp.Any):
    return sum(x.size for x in jax.tree_leaves(params))


def parameters_bytes(params: tp.Any):
    return sum(x.size * x.dtype.itemsize for x in jax.tree_leaves(params))
