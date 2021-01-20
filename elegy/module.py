import functools
import threading
import typing as tp
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from elegy import hooks, utils
from elegy.frozen_dict import FrozenDict
from elegy.types import (
    NoContext,
    ModuleOrderError,
    ParameterCollection,
    Parameters,
    Path,
    Protocol,
    SubmoduleNotRegistered,
)

__all__ = [
    "Module",
    "to_module",
    "add_loss",
    "add_metric",
    "add_summary",
    "next_rng_key",
]

T = tp.TypeVar("T")


class LocalContext(Protocol):
    parent: tp.Optional["Module"]
    module_path: tp.Optional[tp.Dict["Module", Path]]
    inside_call: tp.Optional[bool]
    initializing: tp.Optional[bool]
    module_index: tp.Optional[int]


@dataclass
class _LocalContext(threading.local):
    parent: tp.Optional["Module"]
    module_path: tp.Optional[tp.Dict["Module", Path]]
    inside_call: tp.Optional[bool]
    initializing: tp.Optional[bool]
    module_index: tp.Optional[int]


LOCAL: LocalContext = _LocalContext(
    parent=None,
    module_path=None,
    inside_call=None,
    initializing=None,
    module_index=None,
)


def construct_module(cls, *args, **kwargs) -> "Module":
    module: Module = cls.__new__(cls, *args, **kwargs)
    with instantiation_context(module):
        cls.__init__(module, *args, **kwargs)

    assert module is not None

    if (
        not hasattr(module, "name")
        or not hasattr(module, "_params")
        or not hasattr(module, "_submodules")
    ):
        raise ValueError(
            "Constructing a Module without calling the super constructor "
            "is not supported."
        )

    for key, value in vars(module).items():
        if not key.startswith("_") and leaf_isinstance(value, Module):
            module._submodules.append(key)

            for path, submodule in utils.leaf_paths(value):
                if isinstance(submodule, Module):
                    submodule._path_in_parent[module] = (key,) + path

    return module


class ModuleMeta(ABCMeta):
    def __call__(cls: tp.Type, *args, **kwargs) -> "Module":

        # Set unique on parent when using inside `call`
        if LOCAL.inside_call:
            assert LOCAL.module_index is not None
            assert LOCAL.parent

            index = LOCAL.module_index
            parent = LOCAL.parent

            if len(parent._dynamic_submodules) > index:
                module = getattr(
                    parent,
                    parent._dynamic_submodules[index],
                )
                assert isinstance(module, Module)

                # if not isinstance(module, cls):
                if module.__class__.__name__ != cls.__name__:
                    raise ModuleOrderError(
                        f"Error retrieving module, expected type {cls.__name__}, got {module.__class__.__name__}. "
                        "This is probably due to control flow, you must guarantee that the same amount "
                        "of submodules will be created every time and that their order is the same."
                    )
            else:
                if not LOCAL.initializing:
                    raise ValueError(
                        f"Trying to create module of type'{cls.__name__}' outside of `init`."
                    )

                module = construct_module(cls, *args, **kwargs)

                name = utils.get_unique_name(set(vars(parent)), module.name)
                setattr(parent, name, module)
                parent._submodules.append(name)
                parent._dynamic_submodules.append(name)
                module._path_in_parent[parent] = (name,)

            LOCAL.module_index += 1

            return module
        else:
            return construct_module(cls, *args, **kwargs)


class InitJit(Protocol):
    def __call__(self, *args) -> tp.Tuple[tp.Any, ParameterCollection]:
        ...


class ApplyJit(Protocol):
    def __call__(
        self, parameters: ParameterCollection, *args
    ) -> tp.Tuple[tp.Any, ParameterCollection]:
        ...


class Module(metaclass=ModuleMeta):
    """
    Basic Elegy Module.

    For more information check out the [Module System guide](https://poets-ai.github.io/elegy/guides/module-system/).

    """

    name: str
    dtype: np.dtype
    _params: tp.Dict[str, str]
    _states_initial: tp.List[str]
    _submodules: tp.List[str]
    _dynamic_submodules: tp.List[str]
    _path_in_parent: tp.Dict["Module", Path]

    init_jit: InitJit
    apply_jit: ApplyJit

    __all__ = [
        "__init__",
        "call",
        "add_parameter",
        "get_parameters",
        "set_parameters",
        "reset",
        "init",
    ]

    def __init__(self, name: tp.Optional[str] = None, dtype: np.dtype = tp.Any):
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
        self.dtype = dtype
        self._params = {}
        self._submodules = []
        self._dynamic_submodules = []
        self._path_in_parent = {}

        self._jit_functions()

    def _jit_functions(self):
        def init_jit(*args) -> tp.Tuple[tp.Any, ParameterCollection]:
            return self.init(*args)

        def apply_jit(parameters, *args) -> tp.Tuple[tp.Any, ParameterCollection]:
            if parameters is None:
                raise ValueError("parameters cannot be None with `apply_jit`.")

            return self.apply(parameters, *args)

        self.apply_jit = hooks.jit(apply_jit)
        self.init_jit = hooks.jit(init_jit)

    def __setstate__(self, d):
        self.__dict__ = d
        self._jit_functions()

    def __getstate__(self):
        d = dict(self.__dict__)
        del d["apply_jit"]
        del d["init_jit"]
        return d

    def __call__(self, *args, **kwargs) -> tp.Any:
        """
        Forwards all input arguments to the Module's `call` method and calls
        `elegy.add_summary` on the outputs.
        """

        with call_context(self):

            outputs = self.call(*args, **kwargs)

            if hooks.summaries_active():
                path = get_module_path(self)
                assert path is not None
                hooks.add_summary(path, self, outputs)

            return outputs

    # def __init_subclass__(cls, *args, **kwargs):
    #     super().__init_subclass__(*args, **kwargs)
    #     utils.wraps(cls.call)(cls.init)

    @abstractmethod
    def call(self, *args, **kwargs):
        ...

    def init(self, *args, **kwargs) -> tp.Tuple[tp.Any, ParameterCollection]:
        """
        Initializes the module,
        """

        with init_context():
            y = self(*args, **kwargs)

        return y, self.get_parameters()

    @tp.overload
    def apply(self, parameters: None, *args, **kwargs) -> tp.Any:
        ...

    @tp.overload
    def apply(
        self, parameters: ParameterCollection, *args, **kwargs
    ) -> tp.Tuple[tp.Any, ParameterCollection]:
        ...

    def apply(
        self, __parameters: tp.Optional[ParameterCollection], *args, **kwargs
    ) -> tp.Union[tp.Any, tp.Tuple[tp.Any, ParameterCollection]]:
        if __parameters is not None:
            old_parameters = self.get_parameters()
            self.set_parameters(__parameters)
        else:
            old_parameters = None

        with apply_context():
            y = self(*args, **kwargs)

        if old_parameters is not None:
            new_parameters = self.get_parameters()
            self.set_parameters(old_parameters)

            return y, new_parameters
        else:
            return y

    def add_parameter(
        self,
        name: str,
        initializer: tp.Callable[[], tp.Any],
        collection: str = "parameters",
        regularizer: tp.Optional[tp.Callable[[tp.Any], jnp.ndarray]] = None,
        constraint: tp.Optional[tp.Callable[[tp.Any], tp.Any]] = None,
    ) -> np.ndarray:
        """
        Adds a parameter to the current module. The parameter will only be initialized once and
        will reused afterwards.

        Arguments:
            name: The name of the parameter. It must be unique and no other field/property/method
                of the instance can have that name.
            initializer: A callable that takes not arguments returns the initial value.
            collection: Name of the parameter collection.
            regularizer: Regularizer instance (callable).
            constraint: Constraint instance (callable).

        Returns:
            The value of the parameter.
        """

        if not hasattr(self, name):

            self._params[name] = collection

            initial_value = initializer()

            setattr(self, name, initial_value)

        elif name not in self._params:
            raise ValueError(
                f"Class already contained a property named '{name}', "
                "please use a unique name for the parameter."
            )

        value = getattr(self, name)

        if constraint is not None:
            value = constraint(value)

        if regularizer is not None:
            hooks.add_loss(
                name=utils.get_name(regularizer),
                value=regularizer(value),
            )

        return value

    def add_state(
        self,
        name: str,
        initializer: tp.Callable[[], tp.Any],
        regularizer: tp.Optional[tp.Callable[[tp.Any], jnp.ndarray]] = None,
        constraint: tp.Optional[tp.Callable[[tp.Any], tp.Any]] = None,
    ) -> np.ndarray:
        """
        Adds a parameter to the 'states' collection on the current module. The parameter will only be initialized once and
        will reused afterwards. This is a shortcut for:

        ```python
        self.add_parameter(..., collection="states", ...)
        ```

        Arguments:
            name: The name of the parameter. It must be unique and no other field/property/method
                of the instance can have that name.
            initializer: A callable that takes not arguments returns the initial value.
            regularizer: Regularizer instance (callable).
            constraint: Constraint instance (callable).

        Returns:
            The value of the parameter.
        """
        return self.add_parameter(
            name,
            initializer,
            collection="states",
            regularizer=regularizer,
            constraint=constraint,
        )

    def update_parameter(self, name: str, value: tp.Any) -> None:
        """
        Update a parameter of the current module.

        !!! Note
            Parameters are not updated when `Module.init` is called.

        Arguments:
            name: The name of the parameter to be updated. It must be unique and no other field/property/method
                of the instance can have that name.
            value: The updated value of the state.

        Raises:
            `ValueError` if parameter is not present in current module.
        """

        if not hasattr(self, name):
            raise ValueError(f"Parameter {name} not found in {self}.")

        if is_initializing():
            return

        setattr(self, name, value)

    def add_or_update_parameter(
        self,
        name: str,
        value: tp.Callable[[], tp.Any],
        collection: str = "states",
    ):
        """
        Add a parameter to the current module or update it if it already exists.

        !!! Note
            Parameters are not updated when `Module.init` is called.

        Arguments:
            name: The name of the state. It must be unique and no other field/property/method
                of the instance can have that name.
            value: The updated value of the state.
            collection: Name of the parameter collection.

        Raises:
            `ValueError` if parameter is not present in current module.
        """
        if not hasattr(self, name):
            self.add_parameter(name, lambda: value, collection=collection)
        else:
            self.update_parameter(name, value)

    def get_parameters(
        self,
    ) -> ParameterCollection:
        """
        Recursively collects a dictionary with the parameters of this module
        grouped by collection.
        """

        # find all existing collections
        collections = set()
        tree_exec(lambda module: collections.update(module._params.values()), self)

        # create a dict of collections to the parameters of those collections
        params = {
            collection: module_tree_map(
                lambda module: {
                    key: getattr(module, key)
                    for key, params_collection in module._params.items()
                    if hasattr(module, key) and collection == params_collection
                },
                self,
            )
            for collection in collections
        }

        return params

    def set_parameters(self, parameter_collection: ParameterCollection) -> None:
        """
        Recursively sets all the parameters of this module.
        """

        def f(module: Module, values: tp.Dict[str, tp.Any]):
            for key, value in values.items():
                if key in module._params:
                    setattr(module, key, value)

        for parameters in parameter_collection.values():
            tree_apply(f, self, parameters)

    def reset(self):
        """
        Recursively deletes the parameters and states of this module
        and all submodules within it.
        """

        def clear_module(module: Module):
            for name in module._params:
                delattr(module, name)

        tree_exec(clear_module, self)

    @property
    def submodules(self) -> tp.Dict[str, tp.Any]:
        """
        A dictionary with all submodules contained in this Module.
        """
        return {name: getattr(self, name) for name in self._submodules}


# -------------------------------------------------------------
# hooks
# -------------------------------------------------------------


def get_module_path(module: Module) -> tp.Optional[Path]:
    return (
        LOCAL.module_path[module]
        if module is not None and LOCAL.module_path is not None
        else None
    )


def is_initializing() -> bool:
    return bool(LOCAL.initializing)


# -----------------------------------------------------------------------------
# context managers
# -----------------------------------------------------------------------------


def call_context(module: Module) -> tp.ContextManager[None]:
    return _call_context(module)


@contextmanager
def _call_context(module: Module):

    prev_module = LOCAL.parent
    prev_inside_call = LOCAL.inside_call
    prev_module_index = LOCAL.module_index

    LOCAL.parent = module
    LOCAL.inside_call = True
    LOCAL.module_index = 0

    try:
        if prev_module is not None and prev_module not in module._path_in_parent:
            raise SubmoduleNotRegistered(
                f"Submodule {utils.get_name(module)} not registered in {utils.get_name(prev_module)}, "
                f"this is probably due to some of the following reasons:\n"
                f"- The submodule is being captured by closure and not registered to any field.\n"
                f"- The submodule was set to a field of the parent but "
                f"its contained inside a more complex type which elegy cannot "
                f"inspect, elegy only looks structures of (possibly nested) list, tuple, or dict.\n"
                f"- The submodule was set to a field of the parent by mutating such field after __init__\n\n"
                f"- If non of the previous is true consider this a bug."
                f"Submodule: {module}\n"
                f"Module: {prev_module}\n"
            )

        if hooks.summaries_active():
            if LOCAL.module_path is None:
                raise NoContext(
                    f"Summaries are active but no context for the module is being used, "
                    f"if this happens consider using `Module.init` or `Module.apply` instead "
                    f"of calling the module directly. Got: {module}"
                )
            elif prev_module is None:
                LOCAL.module_path[module] = ()
            else:

                LOCAL.module_path[module] = (
                    LOCAL.module_path[prev_module] + module._path_in_parent[prev_module]
                )
        yield
    finally:
        LOCAL.parent = prev_module
        LOCAL.inside_call = prev_inside_call
        LOCAL.module_index = prev_module_index


def instantiation_context(module: Module) -> tp.ContextManager[None]:
    return _instantiation_context(module)


@contextmanager
def _instantiation_context(module: Module):

    prev_module = LOCAL.parent
    prev_inside_call = LOCAL.inside_call

    LOCAL.inside_call = False

    try:
        yield
    finally:
        LOCAL.parent = prev_module
        LOCAL.inside_call = prev_inside_call


def init_context() -> tp.ContextManager[None]:
    return _init_context()


@contextmanager
def _init_context() -> tp.Iterator[None]:
    prev_initializing = LOCAL.initializing
    prev_module_path = LOCAL.module_path

    LOCAL.initializing = True
    LOCAL.module_path = {}

    try:
        yield
    finally:
        LOCAL.initializing = prev_initializing
        LOCAL.module_path = prev_module_path


def apply_context() -> tp.ContextManager[None]:
    return _apply_context()


@contextmanager
def _apply_context() -> tp.Iterator[None]:
    prev_initializing = LOCAL.initializing
    prev_module_path = LOCAL.module_path

    LOCAL.initializing = False
    LOCAL.module_path = {}

    try:
        yield
    finally:
        LOCAL.initializing = prev_initializing
        LOCAL.module_path = prev_module_path


# ------------------------------------------------------------------------
# utils
# ------------------------------------------------------------------------


def module_tree_map(
    f: tp.Callable[[Module], tp.Dict[str, tp.Any]],
    module: tp.Union[Module, tp.List, tp.Tuple, tp.Dict],
) -> tp.Union[tp.List, tp.Tuple, tp.Dict]:

    if isinstance(module, tp.List):
        return [module_tree_map(f, module) for module in module]
    elif isinstance(module, tp.Tuple):
        return tuple(module_tree_map(f, module) for module in module)
    elif isinstance(module, tp.Dict):
        return {key: module_tree_map(f, module) for key, module in module.items()}
    elif isinstance(module, Module):

        node = f(module)

        for submodule in module._submodules:
            value = module_tree_map(f, getattr(module, submodule))
            # if value:  # drop if empty
            node[submodule] = value

        return node
    else:
        return ()


def tree_apply(
    f: tp.Callable[[Module, tp.Dict[str, tp.Any]], None],
    module: tp.Union[Module, tp.List, tp.Tuple, tp.Dict],
    values: tp.Union[tp.List, tp.Tuple, tp.Dict],
):

    if isinstance(module, tp.List):
        assert isinstance(values, tp.List)

        for module, value in zip(module, values):
            tree_apply(f, module, value)

    elif isinstance(module, tp.Tuple):
        assert isinstance(values, tp.Tuple)

        for module, value in zip(module, values):
            tree_apply(f, module, value)

    elif isinstance(module, tp.Dict):
        assert isinstance(values, tp.Dict)

        for key, value in values.items():
            tree_apply(f, module[key], value)

    elif isinstance(module, Module):
        assert isinstance(values, tp.Dict)

        f(module, values)

        for key, value in values.items():
            if key in module._submodules:
                tree_apply(f, getattr(module, key), value)


def tree_exec(
    f: tp.Callable[[Module], tp.Any],
    module: tp.Union[Module, tp.List, tp.Tuple, tp.Dict],
):

    if isinstance(module, tp.List):

        for module in module:
            tree_exec(f, module)

    elif isinstance(module, tp.Tuple):

        for module in module:
            tree_exec(f, module)

    elif isinstance(module, tp.Dict):

        for key, value in module.items():
            tree_exec(f, module[key])

    elif isinstance(module, Module):
        f(module)

        for key in module._submodules:
            tree_exec(f, getattr(module, key))


def leaf_isinstance(obj: tp.Any, types) -> tp.Type:

    if isinstance(obj, (tp.List, tp.Tuple)) and obj:
        return any(leaf_isinstance(elem, types) for elem in obj)
    elif isinstance(obj, tp.Dict) and obj:
        return any(leaf_isinstance(elem, types) for elem in obj.values())
    else:
        return isinstance(obj, types)


def to_module(f):
    class ToModule(Module):
        def __init__(self, name: tp.Optional[str] = None):
            super().__init__(
                name=utils.lower_snake_case(f.__name__) if name is None else name
            )
            self.call = f

        def call(self, *args, **kwargs):
            ...

    ToModule.__name__ = f.__name__

    return ToModule


def as_initial(name):
    return f"{name}__initial__"
