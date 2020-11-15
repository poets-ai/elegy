import functools
import inspect
import re
import sys
import typing as tp
from enum import Enum
from functools import total_ordering

import jax.numpy as jnp
import numpy as np
import toolz
from deepmerge import always_merger

from elegy.frozen_dict import FrozenDict

if sys.version_info >= (3, 8):
    from typing import Protocol, runtime_checkable
else:
    from typing_extensions import Protocol, runtime_checkable


EPSILON = 1e-7


class TrivialPytree:
    def tree_flatten(self):
        return tuple(vars(self).values()), None

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return cls(*children)


@total_ordering
class Mode(Enum):
    predict = 1
    test = 2
    train = 3

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class Empty:
    pass


class ModuleOrderError(Exception):
    pass


EMPTY = Empty()


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
    rename: tp.Optional[tp.Dict[str, str]] = None,
):
    f_params = get_function_args(f)

    @functools.wraps(f)
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

    apply_kwargs = dict(training=training)
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


def to_static(structure: tp.Any) -> tp.Any:

    if isinstance(structure, (tp.Dict, FrozenDict)):
        return tuple((k, to_static(v)) for k, v in structure.items())
    elif isinstance(structure, (tp.List, tp.Tuple)):
        return tuple(to_static(v) for v in structure)
    else:
        return structure
