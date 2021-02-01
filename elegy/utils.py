import functools
import hashlib
import inspect
import os
import re
import shutil
import sys
import typing as tp
import urllib
from functools import total_ordering

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np
import toolz

from elegy import types


def maybe_expand_dims(a: np.ndarray, b: np.ndarray) -> tp.Tuple[np.ndarray, np.ndarray]:
    assert np.prod(a.shape) == np.prod(b.shape)

    if a.ndim < b.ndim:
        a = a[..., None]

    if b.ndim < a.ndim:
        b = b[..., None]

    return a, b


def wraps(f, docs: bool = True):
    assignments = ("__annotations__",)

    if docs:
        assignments += ("__doc__",)

    return functools.wraps(f, assigned=assignments, updated=())


def get_signature_f_recursive(f: tp.Callable) -> tp.Callable:

    if hasattr(f, "_signature_f"):
        return get_signature_f_recursive(f._signature_f)
    else:
        return f


def inject_dependencies(
    f: tp.Callable,
    signature_f: tp.Optional[tp.Callable] = None,
    rename: tp.Optional[tp.Dict[str, str]] = None,
):
    if signature_f is None:
        signature_f = f

    signature_f = get_signature_f_recursive(signature_f)
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
    states: types.States,
    initializing: bool,
    mode: types.Mode,
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
        initializing=initializing,
        net_params=states.net_params,
        net_states=states.net_states,
        rng=states.rng,
        states=states,
        mode=mode,
        training=(mode == types.Mode.train),
    )
    apply_kwargs.update(kwargs)

    return args, apply_kwargs


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


def _leaf_paths(
    path: types.Path, inputs: tp.Any
) -> tp.Iterable[tp.Tuple[types.Path, tp.Any]]:

    if isinstance(inputs, (tp.Tuple, tp.List)):
        for i, value in enumerate(inputs):
            yield from _leaf_paths(path + (i,), value)
    elif isinstance(inputs, tp.Dict):
        for name, value in inputs.items():
            yield from _leaf_paths(path + (name,), value)
    else:
        yield (path, inputs)


def leaf_paths(inputs: tp.Any) -> tp.List[tp.Tuple[types.Path, tp.Any]]:
    return list(_leaf_paths((), inputs))


def _flatten_names(
    path: types.Path, inputs: tp.Any
) -> tp.Iterable[tp.Tuple[types.Path, tp.Any]]:

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


def parameters_count(params: tp.Any) -> int:
    leaves = (jnp.asarray(x) for x in jax.tree_leaves(params))
    return sum(x.size for x in leaves)


def parameters_bytes(params: tp.Any) -> int:
    leaves = (jnp.asarray(x) for x in jax.tree_leaves(params))
    return sum(x.size * x.dtype.itemsize for x in leaves)


def download_file(url, cache="~/.elegy/downloads", sha256=None):
    if cache.startswith("~/"):
        cache = os.path.join(os.path.expanduser("~"), cache[2:])
    cachefilename = os.path.basename(url)
    cachefilename = cachefilename[: cachefilename.find("?")]
    cachefilename = os.path.join(cache, cachefilename)

    if not os.path.exists(cachefilename):
        print(f"Downloading {url}")
        filename, _ = urllib.request.urlretrieve(url)
        if sha256 is not None:
            filehash = hashlib.sha256(open(filename, "rb").read()).hexdigest()
            if sha256 != filehash:
                raise RuntimeError("Downloaded file has an incorrect hash")
        os.makedirs(os.path.dirname(cachefilename), exist_ok=True)
        shutil.move(filename, cachefilename)

    return cachefilename


def merge_params(a: tp.Any, b: tp.Any):

    if isinstance(a, dict) and isinstance(b, dict):
        return {
            key: a[key]
            if key not in b
            else b[key]
            if key not in a
            else merge_params(a[key], b[key])
            for key in set(a) | set(b)
        }
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise ValueError(
                f"Cannot merge two lists of different lengths:\na={a}\nb={b}"
            )
        return [merge_params(a, b) for a, b in zip(a, b)]
    elif isinstance(a, tuple) and isinstance(b, tuple):
        if len(a) != len(b):
            raise ValueError(
                f"Cannot merge two tuples of different lengths:\na={a}\nb={b}"
            )
        return tuple(merge_params(a, b) for a, b in zip(a, b))
    else:
        raise ValueError(f"Cannot merge structs:\na={a}\nb={b}")


def get_path_params(path: types.Path, params: tp.Any) -> tp.Any:
    for key in path:
        try:
            params = params[key]
        except BaseException as e:
            return None

    return params


def none_or_uninitialized(x: tp.Any) -> bool:
    return x is None or isinstance(x, types.Uninitialized)


# def merge_collections(collections: types.ParameterCollection) -> tp.Dict[str, tp.Any]:
#     output_parameters = {}

#     for collection, values in collections.items():
#         params = jax.tree_map(lambda x: types.Parameter(x, collection), values)
#         output_parameters = merge_params(params, output_parameters)

#     assert isinstance(output_parameters, dict)

#     return output_parameters


def get_parameter(collections: types.ParameterCollection, name: str) -> types.Parameter:

    parameters = [
        (collection, parameters[name])
        for collection, parameters in collections.items()
        if name in parameters
    ]

    if len(parameters) == 0:
        raise ValueError(f"No parameters named {name} in collections {collections}")
    elif len(parameters) >= 2:
        raise ValueError(
            f"Multiple parameters named {name} in collections {collections}"
        )

    [(collection, value)] = parameters

    return types.Parameter(collection=collection, value=value)


def get_submodule_colletions(
    collections: types.ParameterCollection, name: str
) -> types.ParameterCollection:
    return {
        collection: parameters[name]
        for collection, parameters in collections.items()
        if name in parameters
    }


def split_into_collections(
    parameters: tp.Dict[str, tp.Any]
) -> types.ParameterCollection:
    all_collections = set()

    def find_collections(parameter: types.Parameter):
        all_collections.add(parameter.collection)

    jax.tree_map(find_collections, parameters)

    return {
        collection: unwrap_filter(
            lambda p: p.collection == collection,
            parameters,
        )
        for collection in all_collections
    }


def unwrap_filter(
    f: tp.Callable[[types.Parameter], bool], parameters: tp.Dict[str, tp.Any]
) -> tp.Dict[str, tp.Any]:
    outputs = {}

    for name, parameter in parameters.items():

        if isinstance(parameter, types.Parameter):
            if f(parameter):
                outputs[name] = parameter.value
        else:
            outputs[name] = unwrap_filter(f, parameter)

    return outputs


def plot_history(history):
    import matplotlib.pyplot as plt

    keys = [key for key in history.history.keys() if not key.startswith("val_")]
    n_plots = len(keys)

    figure = plt.figure(figsize=(14, 24))

    # for i, key in enumerate(list(history.history.keys())[:n_plots]):
    for i, key in enumerate(keys):
        if key == "size":
            continue

        metric = history.history[key]

        plt.subplot(n_plots, 1, i + 1)
        plt.plot(metric, label=f"Training {key}")

        try:
            val_metric = history.history[f"val_{key}"]
            plt.plot(val_metric, label=f"Validation {key}")
            title = f"Training and Validation {key}"
        except KeyError:
            title = f"Training {key}"

        plt.legend(loc="lower right")
        plt.ylabel(key)
        plt.title(title)

    plt.show()
