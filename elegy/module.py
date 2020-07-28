import typing as tp
from abc import abstractmethod

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from elegy import utils
from elegy import hooks


class Defered(tp.NamedTuple):
    """
    A module that defers the construction of Module types.

    Defered is a generic wrapper over a type, it takes any callable `Type` plus the `args` 
    and `kwargs` arguments accepted by the types constructor, and let you construct
    and instance later. Defered has some practical usecases:

    **1.** It allows you to construct objects that hold the arguments for `Module`s while
    retaining their callable behaviour. This allows go around Haiku's restriction of 
    not being able instantiate Modules outside of a `transform` and is its primary
    usecase as it allows you to pass parametrized Modules to Elegy's Model constructor:
    

    ```python
    class SomeModule(elegy.Module):
        def __init__(self, a, b):
            ...

        def call(self, x):
            ...

    model = elegy.Model(
        module=SomeModule.defer(a=1, b=2),
        ...
    )
    ```

    **2.** It can also serve as a mechanism for reusing hyperparameters since you can define
    them once and an call them mutliple times:

    ```python
    some_defered = SomeModule.defer(a=1, b=2)

    class OtherModule(elegy.Module):
        def call(self, x):
            x = some_defered(x)
            x = some_defered(x)
            return x
    ```
    Note that here we are sharing hyperparameters but **not** weights since the previous is
    equivalent to:

    ```python
    class OtherModule(elegy.Module):
        def call(self, x):
            x = SomeModule(a=1, b=2)(x)
            x = SomeModule(a=1, b=2)(x)
            return x
    ```
    
    To enable weight sharing from a Defered object just get hold an instance of the `Module`
    and use Haiku's normal weight sharing mechanism:

    ```python
    class OtherModule(elegy.Module):
        def call(self, x):
            some_module = some_defered.get_instance()
            x = some_module(x)
            x = some_module(x)
            return x
    ```
    This is equivalent to:

    ```python
    class OtherModule(elegy.Module):
        def call(self, x):
            some_module = SomeModule(a=1, b=2)
            x = some_module(x)
            x = some_module(x)
            return x
    ```
    and is probably how you should do parameter sharing most of the time.
    """

    csl: tp.Type
    args: tp.Tuple
    kwargs: tp.Dict[str, tp.Any]

    def __call__(self, *args, **kwargs):
        """
        Creates an instance of the wrapped type / Module and forwards
        all inputs to the Module's `__call__` method while also providing
        dependency injection (optional argument not required by the Module 
        will be ignored).
        """
        return utils.inject_dependencies(self.csl(*self.args, **self.kwargs))(
            *args, **kwargs
        )

    def get_instance(self) -> tp.Any:
        """
        Construct a new instance of the type wrapper by Defered.
        """
        return self.csl(*self.args, **self.kwargs)


class Deferable:
    @classmethod
    def defer(cls, *args, **kwargs) -> Defered:
        """
        """
        return Defered(cls, args=args, kwargs=kwargs)


class Module(hk.Module, Deferable):
    """
    Basic Elegy Module. Its a thin wrapper around `hk.Module` that
    add custom functionalities related to Elegy.
    """

    def __init__(self, name: tp.Optional[str] = None):
        """
        Initializes the current module with the given name.

        Subclasses should call this constructor before creating other modules or
        variables such that those modules are named correctly.

        Arguments:
            name: An optional string name for the class. Must be a valid Python
                identifier. If ``name`` is not provided then the class name for the
                current instance is converted to ``lower_snake_case`` and used instead.
        """
        super().__init__(name=name)

        self.call = utils.inject_dependencies(self.call)

    def __call__(self, *args, **kwargs):
        """
        Forwards all input arguments to the Module's `call` method and calls
        `elegy.add_summary` on the outputs.
        """
        outputs = self.call(*args, **kwargs)

        hooks.add_summary(None, self.__class__.__name__, outputs)

        return outputs

    @abstractmethod
    def call(self, *args, **kwargs):
        ...

    @classmethod
    def defer(cls, *args, **kwargs) -> Defered:
        """
        Creates a [`Defered`][elegy.module.Defered] instance for this class.

        All arguments (positional and keyword) passed to `defer` must be accepted by
        the `Module`'s constructor.
        """
        return Defered(cls, args=args, kwargs=kwargs)
