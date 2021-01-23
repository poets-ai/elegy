from elegy.types import Index, Path, States
import functools
import inspect
import re
import sys
import typing as tp
from functools import total_ordering
import urllib, hashlib, shutil, os

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np
import toolz
from deepmerge import always_merger


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


def inject_dependencies(
    f: tp.Callable,
    signature_f: tp.Optional[tp.Callable] = None,
    rename: tp.Optional[tp.Dict[str, str]] = None,
):
    if signature_f is not None:
        pass
    elif hasattr(f, "_signature_f") and f._signature_f is not None:
        signature_f = f._signature_f
    elif signature_f is None:
        signature_f = f
    assert signature_f is not None

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
