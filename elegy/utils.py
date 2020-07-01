import typing as tp
import inspect


class DIFunction(tp.NamedTuple):
    f: tp.Callable
    fargs: tp.List[str]

    @classmethod
    def create(cls, f: tp.Callable) -> "DIFunction":
        return cls(f, get_function_args(f))

    def __call__(self, *args, **kwargs):
        # print("f", self.f)
        # print("fargs", self.fargs)
        # print("args", len(args))
        # print("kwargs", list(kwargs.keys()))
        return self.f(**{arg: kwargs[arg] for arg in self.fargs})


def get_function_args(f) -> tp.List[str]:
    return list(inspect.signature(f).parameters.keys())


if __name__ == "__main__":

    def f(a, b, d, c=1):
        return a + b + c + d

    g = DIFunction.create(f)

    print(g(a=1, b=2, c=3, d=4))
