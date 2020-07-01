import typing as tp
import inspect


class DIFunction(tp.NamedTuple):
    f: tp.Callable
    f_params: tp.List[str]

    @classmethod
    def create(cls, f: tp.Callable) -> "DIFunction":
        return cls(f, get_function_args(f))

    def __call__(self, *args, **kwargs):
        n_args = len(args)
        arg_names = self.f_params[:n_args]
        kwarg_names = self.f_params[n_args:]

        return self.f(
            *args, **{arg: kwargs[arg] for arg in kwarg_names if arg not in arg_names}
        )


def get_function_args(f) -> tp.List[str]:
    return list(inspect.signature(f).parameters.keys())
