import inspect
import typing as tp

import jax.numpy as jnp
import numpy as np


class DIFunction(tp.NamedTuple):
    f: tp.Callable
    f_params: tp.List[inspect.Parameter]
    rename: tp.Optional[tp.Dict[str, str]]

    @classmethod
    def create(
        cls, f: tp.Callable, rename: tp.Optional[tp.Dict[str, str]] = None
    ) -> "DIFunction":
        return cls(f, get_function_args(f), rename)

    def __call__(self, *args, **kwargs):
        n_args = len(args)
        arg_names = [arg.name for arg in self.f_params[:n_args]]
        kwarg_names = [arg.name for arg in self.f_params[n_args:]]

        if self.rename:
            for old, new in self.rename.items():
                if old in kwargs:
                    kwargs[new] = kwargs.pop(old)

        return self.f(
            *args,
            **(
                kwargs
                if any(
                    arg.kind == inspect.Parameter.VAR_KEYWORD for arg in self.f_params
                )
                else {arg: kwargs[arg] for arg in kwarg_names if arg not in arg_names}
            )
        )


def get_function_args(f) -> tp.List[inspect.Parameter]:
    return list(inspect.signature(f).parameters.values())


def get_input_args(
    x: tp.Union[np.ndarray, jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple], y: tp.Any,
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

    apply_kwargs = dict(y=y)
    apply_kwargs.update(kwargs)

    return args, kwargs

