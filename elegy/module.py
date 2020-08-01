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

        def __apply__(self, x):
            ...

    model = elegy.Model(
        module=SomeModule.defer(a=1, b=2),
        ...
    )
    ```

    **2.** It can also serve as a mechanism for reusing hyperparameters since you can define
    them once and an __apply__ them mutliple times:

    ```python
    some_defered = SomeModule.defer(a=1, b=2)

    class OtherModule(elegy.Module):
        def __apply__(self, x):
            x = some_defered(x)
            x = some_defered(x)
            return x
    ```
    Note that here we are sharing hyperparameters but **not** weights since the previous is
    equivalent to:

    ```python
    class OtherModule(elegy.Module):
        def __apply__(self, x):
            x = SomeModule(a=1, b=2)(x)
            x = SomeModule(a=1, b=2)(x)
            return x
    ```
    
    To enable weight sharing from a Defered object just get hold an instance of the `Module`
    and use Haiku's normal weight sharing mechanism:

    ```python
    class OtherModule(elegy.Module):
        def __apply__(self, x):
            some_module = some_defered.get_instance()
            x = some_module(x)
            x = some_module(x)
            return x
    ```
    This is equivalent to:

    ```python
    class OtherModule(elegy.Module):
        def __apply__(self, x):
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


class Module:
    """
    Basic Elegy Module. Its a thin wrapper around `hk.Module` that
    add custom functionalities related to Elegy.
    """

    name: str
    _params: tp.List[str]
    _initialized: bool

    def __init__(self, name: tp.Optional[str] = None):
        """
        Initializes the current module with the given name.

        Subclasses should call this constructor before creating other modules or
        variables such that those modules are named correctly.

        Arguments:
            name: An optional string name for the class. Must be a valid elsePython
                identifier. If ``name`` is not provided then the class name for the
                current instance is converted to ``lower_snake_case`` and used instead.
        """
        self.name = name if name else utils.lower_snake_case(self.__class__.__name__)
        self._initialized = False
        self._params = []

    def __call__(self, *args, **kwargs) -> tp.Any:
        """
        Forwards all input arguments to the Module's `__apply__` method and calls
        `elegy.add_summary` on the outputs.
        """
        outputs = self.__apply__(*args, **kwargs)

        hooks.add_summary(None, self.__class__.__name__, outputs)

        self._initialized = True

        return outputs

    @abstractmethod
    def __apply__(self, *args, **kwargs):
        ...

    def get_parameter(
        self,
        name: str,
        shape: tp.Sequence[int],
        dtype: tp.Any = jnp.float32,
        initializer: tp.Callable[[tp.Sequence[int], tp.Any], np.ndarray] = None,
    ) -> np.ndarray:

        if name not in self._params:
            setattr(self, name, initializer(shape, dtype))
            self._params.append(name)

        return getattr(self, name)

    @property
    def parameters(self):
        return get_parameters(self)

    @parameters.setter
    def parameters(self, values: tp.Dict):
        return set_parameters(self, values)


def get_parameters(
    module: tp.Union[Module, tp.List, tp.Tuple, tp.Dict]
) -> tp.Union[tp.List, tp.Tuple, tp.Dict]:

    if isinstance(module, tp.List):
        return [get_parameters(module) for module in module]
    elif isinstance(module, tp.Tuple):
        return tuple(get_parameters(module) for module in module)
    elif isinstance(module, tp.Dict):
        return {key: get_parameters(module) for key, module in module.items()}
    else:
        return dict(
            **{key: getattr(module, key) for key in module._params},
            **{
                key: get_parameters(value)
                for key, value in vars(module).items()
                if key != "_params" and leaf_isinstance(value, Module)
            },
        )


def set_parameters(
    module: tp.Union[Module, tp.List, tp.Tuple, tp.Dict],
    values: tp.Union[tp.List, tp.Tuple, tp.Dict],
):

    if isinstance(module, tp.List):
        assert isinstance(values, tp.List)

        for module, value in zip(module, values):
            set_parameters(module, value)

    elif isinstance(module, tp.Tuple):
        assert isinstance(values, tp.Tuple)

        for module, value in zip(module, values):
            set_parameters(module, value)

    elif isinstance(module, tp.Dict):
        assert isinstance(values, tp.Dict)

        for key, value in values.items():
            set_parameters(module[key], value)

    else:
        assert isinstance(values, tp.Dict)

        for key, value in values.items():
            if key in module._params:
                setattr(module, key, value)
            else:
                set_parameters(getattr(module, key), value)


def leaf_isinstance(obj: tp.Any, types) -> tp.Type:

    if isinstance(obj, (tp.List, tp.Tuple)):
        return leaf_isinstance(obj[0], types)
    elif isinstance(obj, tp.Dict):
        return leaf_isinstance(list(obj.values())[0], types)
    else:
        return isinstance(obj, types)
