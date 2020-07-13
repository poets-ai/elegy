import inspect


class A:
    def __init__(self, a, b, c=1):
        ...


def get_function_args(f):
    return list(inspect.signature(f).parameters.values())


print(get_function_args(A))


abc = lambda x: x


print(abc.__name__)
