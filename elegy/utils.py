from dataclasses import dataclass
import functools
import inspect
import sys
import typing as tp

import jax.numpy as jnp
import numpy as np

if sys.version_info >= (3, 8):
    from typing import Protocol, runtime_checkable
else:
    from typing_extensions import Protocol, runtime_checkable


EPSILON = 1e-7


def inject_dependencies(
    f: tp.Callable, rename: tp.Optional[tp.Dict[str, str]] = None,
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
    x: tp.Union[np.ndarray, jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple],
    is_training: bool,
) -> tp.Tuple[tp.Tuple, tp.Mapping[str, tp.Any]]:

    if isinstance(x, tp.Tuple):
        args = x
        kwargs = {}
    elif isinstance(x, tp.Mapping):
        args = ()
        kwargs = x
    else:
        args = (x,)
        kwargs = {}

    apply_kwargs = dict(is_training=is_training)
    apply_kwargs.update(kwargs)

    return args, kwargs


class Defered:
    f: tp.Union[tp.Callable, tp.Type]
    args: tp.Tuple
    kwargs: tp.Dict[str, tp.Any]

    def __init__(self, f: tp.Callable, *args, **kwargs):
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.f(*self.args, **self.kwargs)


class Deferable:
    @classmethod
    def defer(cls, *args, **kwargs) -> Defered:
        return Defered(cls, *args, **kwargs)
