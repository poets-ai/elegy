import typing as tp
import inspect


class DIFunction(tp.NamedTuple):
    f: tp.Callable
    f_params: tp.List[str]
    rename: tp.Optional[tp.Dict[str, str]]

    @classmethod
    def create(
        cls, f: tp.Callable, rename: tp.Optional[tp.Dict[str, str]] = None
    ) -> "DIFunction":
        return cls(f, get_function_args(f), rename)

    def __call__(self, *args, **kwargs):
        n_args = len(args)
        arg_names = self.f_params[:n_args]
        kwarg_names = self.f_params[n_args:]

        if self.rename:
            for old, new in self.rename.items():
                if old in kwargs:
                    kwargs[new] = kwargs.pop(old)

        return self.f(
            *args, **{arg: kwargs[arg] for arg in kwarg_names if arg not in arg_names}
        )


def get_function_args(f) -> tp.List[str]:
    return list(inspect.signature(f).parameters.keys())
