import functools
import threading
import typing as tp
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np

from elegy import hooks, utils
from elegy.types import (
    ApplyJit,
    InitJit,
    JitCallable,
    MissingParameter,
    NoContext,
    ModuleOrderError,
    Parameter,
    ParameterCollection,
    Parameters,
    Path,
    Protocol,
    RNGSeq,
    ShapeMismatch,
    SubmoduleNotRegistered,
)

__all__ = [
    "Module",
    "to_module",
    "add_loss",
    "add_metric",
    "add_summary",
    "next_key",
]

T = tp.TypeVar("T")


class LocalContext(Protocol):
    parent: tp.Optional["Module"]
    module_path: tp.Optional[tp.Dict["Module", Path]]
    inside_call: tp.Optional[bool]
    module_index: tp.Optional[int]
    scope: tp.Optional[tp.Dict[str, tp.Any]]


@dataclass
class _LocalContext(threading.local):
    parent: tp.Optional["Module"]
    module_path: tp.Optional[tp.Dict["Module", Path]]
    inside_call: tp.Optional[bool]
    module_index: tp.Optional[int]
    scope: tp.Optional[tp.Dict[str, tp.Any]]


LOCAL: LocalContext = _LocalContext(
    parent=None,
    module_path=None,
    inside_call=None,
    module_index=None,
    scope=None,
)


def construct_module(cls, *args, **kwargs) -> "Module":
    # call constructor
    module: Module = cls.__new__(cls, *args, **kwargs)
    with instantiation_context(module):
        cls.__init__(module, *args, **kwargs)

    assert module is not None

    if not hasattr(module, "_submodules"):
        raise ValueError(
            "Constructing a Module without calling the super constructor "
            "is not supported."
        )

    # register submodules created in __init__
    for key, value in vars(module).items():
        if not key.startswith("_") and leaf_isinstance(value, Module):

            for path, submodule in utils.leaf_paths(value):
                if isinstance(submodule, Module):
                    path = (key,) + path
                    name = "/".join(map(str, path))
                    module._submodules[name] = submodule
                    module._submodule_name[submodule] = name

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
                module = parent._dynamic_submodules[index]

                assert isinstance(module, Module)

                # if not isinstance(module, cls):
                if module.__class__.__name__ != cls.__name__:
                    raise ModuleOrderError(
                        f"Error retrieving module, expected type {cls.__name__}, got {module.__class__.__name__}. "
                        "This is probably due to control flow, you must guarantee that the same amount "
                        "of submodules will be created every time and that their order is the same."
                    )
            else:
                # if not LOCAL.initializing:
                #     raise ValueError(
                #         f"Trying to create module of type'{cls.__name__}' outside of `init`."
                #     )

                module = construct_module(cls, *args, **kwargs)

                name = utils.get_unique_name(set(vars(parent)), module.name)

                parent._submodules[name] = module
                parent._submodule_name[module] = name
                parent._dynamic_submodules.append(module)

            LOCAL.module_index += 1

            return module
        else:
            return construct_module(cls, *args, **kwargs)


class Module(metaclass=ModuleMeta):
    """
    Basic Elegy Module.

    For more information check out the [Module System guide](https://poets-ai.github.io/elegy/guides/module-system/).

    """

    name: str
    dtype: np.dtype
    _states_initial: tp.List[str]
    _submodules: tp.Dict[str, "Module"]
    _submodule_name: tp.Dict["Module", str]
    _dynamic_submodules: tp.List["Module"]
    _default_scope: tp.Dict[str, tp.Any]

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

    def __init__(self, name: tp.Optional[str] = None, dtype: tp.Any = jnp.float32):
        """
        Initializes the current module with the given name.

        Subclasses should call this constructor before creating other modules or
        variables such that those modules are named correctly.

        Arguments:
            name: An optional string name for the class. If ``name`` is not provided then the class name for the
                current instance is converted to ``lower_snake_case`` and used instead.
        """
        self.name = name if name else utils.lower_snake_case(self.__class__.__name__)
        self.dtype = dtype
        self._submodules = {}
        self._submodule_name = {}
        self._dynamic_submodules = []
        self._signature_f = self.call
        self._default_scope = {}

        self._jit_functions()

    def call_jit(self, *args) -> tp.Any:
        collections = self.get_parameters()
        y, collections = self.apply_jit(parameters=collections)(*args)
        self.set_parameters(collections)
        return y

    def _jit_functions(self):
        # ------------------------------
        # init
        # ------------------------------
        def init_jit_raw(*args):
            return self.init()(*args)

        init_jit: JitCallable = hooks.jit(init_jit_raw)

        def init_jit_wrapper(
            rng: tp.Optional[RNGSeq] = None,
            **hooks_kwargs,
        ) -> JitCallable:
            def init_jit_callable(
                *args,
            ) -> tp.Tuple[tp.Any, ParameterCollection]:
                with hooks.update_context(rng=rng, **hooks_kwargs):
                    y, collections = init_jit(*args)

                    # set parameters to avoid traced arrays
                    self.set_parameters(collections)

                    return y, collections

            init_jit_callable._signature_f = self.call

            return init_jit_callable

        # ------------------------------
        # apply
        # ------------------------------
        def apply_jit_raw(parameters, *args):
            return self.apply(parameters)(*args)

        apply_jit = hooks.jit(apply_jit_raw)

        def apply_jit_wrapper(
            parameters: ParameterCollection,
            *,
            rng: tp.Optional[RNGSeq] = None,
            **hooks_kwargs,
        ) -> JitCallable:
            def apply_jit_callable(
                *args,
            ) -> tp.Tuple[tp.Any, ParameterCollection]:
                with hooks.update_context(rng=rng, **hooks_kwargs):
                    return apply_jit(parameters, *args)

            apply_jit_callable._signature_f = self.call

            return apply_jit_callable

        # ------------------------------
        # assign functions
        # ------------------------------
        self.init_jit = init_jit_wrapper
        self.apply_jit = apply_jit_wrapper

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

    @abstractmethod
    def call(self, *args, **kwargs):
        ...

    def add_summary(self, name: str, f: tp.Any, value: tp.Any):
        if hooks.summaries_active():
            path = get_module_path(self) + (name,)
            assert path is not None
            hooks.add_summary(path, f, value)

    def init(
        self,
        *,
        rng: tp.Optional[RNGSeq] = None,
        **hooks_kwargs,
    ) -> tp.Callable[..., tp.Tuple[tp.Any, ParameterCollection]]:
        """
        Initializes the module,
        """

        def init_callable(*args, **kwargs) -> tp.Tuple[tp.Any, ParameterCollection]:

            with hooks.update_context(
                rng=rng,
                initializing=True,
                **hooks_kwargs,
            ), scope_context(None):
                y = self(*args, **kwargs)

            return y, self.get_parameters()

        init_callable._signature_f = self.call

        return init_callable

    def apply(
        self,
        parameters: ParameterCollection,
        rng: tp.Optional[RNGSeq] = None,
        **hooks_kwargs,
    ) -> tp.Callable[..., tp.Union[tp.Any, tp.Tuple[tp.Any, ParameterCollection]]]:
        """
        Call the module.
        """

        def apply_callable(*args, **kwargs) -> tp.Tuple[tp.Any, ParameterCollection]:

            with hooks.update_context(
                initializing=False,
                rng=rng,
                **hooks_kwargs,
            ), scope_context(parameters):
                y = self(*args, **kwargs)

                scope = LOCAL.scope
                assert scope is not None

            return y, scope

        apply_callable._signature_f = self.call

        return apply_callable

    def add_parameter(
        self,
        name: str,
        initializer: tp.Callable[[], tp.Any],
        collection: tp.Optional[str] = None,
        trainable: bool = True,
        regularizer: tp.Optional[tp.Callable[[tp.Any], jnp.ndarray]] = None,
        constraint: tp.Optional[tp.Callable[[tp.Any], tp.Any]] = None,
    ) -> tp.Any:
        """
        Adds a parameter to the current module. The parameter will only be initialized once and
        will reused afterwards.

        Arguments:
            name: The name of the parameter. It must be unique and no other field/property/method
                of the instance can have that name.
            initializer: A callable that takes not arguments returns the initial value.
            collection: Optional name of the parameter collection, if not defined it will be se to
                `"parameters"` if `trainable=True` else it will be set to `"states"`.
            trainable: Specify whether this parameter should be added to the default trainable `"parameters"`
                collection or to the default non-trainable `"states"` collection. If collection is
                passed this parameter will ignored.
            regularizer: Regularizer instance (callable).
            constraint: Constraint instance (callable).

        Returns:
            The value of the parameter.
        """

        if collection is None:
            collection = "parameters" if trainable else "states"

        assert LOCAL.scope is not None

        if name not in LOCAL.scope:

            if not hooks.is_initializing():
                raise ValueError(f"Cannot add parameter {name} outside `init`")

            if name in self._default_scope:
                parameter = self._default_scope[name]
                assert isinstance(parameter, Parameter)
                assert collection == parameter.collection
                initial_value = parameter.value

            else:
                initial_value = initializer()
                parameter = Parameter(initial_value, collection)

                self._default_scope[name] = parameter

            LOCAL.scope[name] = parameter

        value = LOCAL.scope[name].value

        if constraint is not None:
            value = constraint(value)

        if regularizer is not None:
            hooks.add_loss(
                name=utils.get_name(regularizer),
                value=regularizer(value),
            )

        return value

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

        assert LOCAL.scope is not None

        if name not in LOCAL.scope:
            raise ValueError(f"Parameter {name} not found in {self}.")

        if hooks.is_initializing():
            return

        parameter = LOCAL.scope[name]
        assert isinstance(parameter, Parameter)
        parameter.value = value

    def add_or_update_parameter(
        self,
        name: str,
        value: tp.Callable[[], tp.Any],
        collection: tp.Optional[str] = None,
        trainable: bool = True,
    ):
        """
        Add a parameter to the current module or update it if it already exists.

        !!! Note
            Parameters are not updated when `Module.init` is called.

        Arguments:
            name: The name of the state. It must be unique and no other field/property/method
                of the instance can have that name.
            value: The updated value of the state.
            collection: Optional name of the parameter collection, if not defined it will be se to
                `"parameters"` if `trainable=True` else it will be set to `"states"`.
            trainable: Specify whether this parameter should be added to the default trainable `"parameters"`
                collection or to the default non-trainable `"states"` collection. If collection is
                passed this parameter will ignored.

        Raises:
            `ValueError` if parameter is not present in current module.
        """
        assert LOCAL.scope is not None

        if name not in LOCAL.scope:
            self.add_parameter(
                name, lambda: value, trainable=trainable, collection=collection
            )
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

    def set_parameters(
        self,
        parameter_collection: ParameterCollection,
        check_missing: tp.Optional[bool] = True,
        check_shapes: tp.Optional[bool] = False,
        ignore_on_error: tp.Optional[tp.Union[bool, str]] = False,
    ) -> None:
        """
        Recursively sets all the parameters of this module.

        Arguments:
            values: A nested dictionary as returned by get_parameters()
            check_missing: If True will check if the new set of parameters is complete. Default: False
            check_shapes: If True (Default) will check if the new parameters have the same shape as the old ones
            ignore_on_error: If True will set only parameters that have a matching shape, ignoring mismatches and missing parameters.
                             If "silent" will not even print a warning.
                             If False (Default) will raise a ValueError on shape mismatch or missing parameter.
        """

        def check_shapes_f(module: Module, values: tp.Dict[str, tp.Any]):
            for key, value in list(values.items()):
                if key in module._params:
                    if not hasattr(module, key):
                        # key in _params but not in module
                        # this can happen after a .reset()
                        # ignore
                        continue
                    prevshape = np.shape(getattr(module, key))
                    newshape = np.shape(value)
                    if prevshape != newshape:
                        errormsg = f"Shape mismatch on parameter {key} in module {module.name}: {prevshape} (old) vs {newshape} (new)."
                        if ignore_on_error:
                            if not ignore_on_error == "silent":
                                print(errormsg + " Ignoring")
                            # ignore by removing from new parameters
                            del values[key]
                        else:
                            raise ValueError(errormsg)
                elif key not in module._submodules and not ignore_on_error == "silent":
                    print(f"Parameter {key} not found in module {module.name}")

            if check_missing:
                missing = [
                    param
                    for param in list(module._params.keys()) + module._submodules
                    if param not in values
                ]
                if len(missing):
                    errormsg = f"Missing parameters in module {module.name}: {missing}"
                    if ignore_on_error == True:
                        print(errormsg)
                    else:
                        raise ValueError(errormsg)

        def set_f(module: Module, values: tp.Dict[str, tp.Any]):
            for key, value in values.items():
                if key in module._params:
                    setattr(module, key, value)

        # first perform the check to avoid setting some parameters then encountering invalid ones
        if check_shapes or check_missing:
            parameters = {}

            for params in parameter_collection.values():
                parameters = utils.merge_params(parameters, params)

            # shape check modifies values, make a copy to keep the original ones untouched
            tree_apply(check_shapes_f, self, parameters)

            # set values
            tree_apply(set_f, self, parameters)

        else:
            # set values
            for parameters in parameter_collection.values():
                tree_apply(set_f, self, parameters)

    def has_parameter(self, name: str) -> bool:
        return hasattr(self, name)

    def reset(self):
        """
        Recursively deletes the parameters and states of this module
        and all submodules within it.
        """

        def clear_module(module: Module):
            for name in module._params:
                if hasattr(module, name):
                    delattr(module, name)

        tree_exec(clear_module, self)


# -------------------------------------------------------------
# hooks
# -------------------------------------------------------------


def get_module_path(module: tp.Optional[Module] = None) -> tp.Optional[Path]:
    if module is None:
        module = LOCAL.parent

    return (
        LOCAL.module_path[module]
        if module is not None and LOCAL.module_path is not None
        else None
    )


def get_module_path_str(module: Module) -> tp.Optional[str]:
    return (
        "/".join(map(str, LOCAL.module_path[module]))
        if module is not None and LOCAL.module_path is not None
        else None
    )


def collect_parameters() -> ParameterCollection:
    assert LOCAL.scope is not None

    collections = set(p.collection for p in jax.tree_leaves(LOCAL.scope))

    # TODO: group params by collection


# -----------------------------------------------------------------------------
# context managers
# -----------------------------------------------------------------------------


def scope_context(scope: tp.Optional[tp.Dict[str, tp.Any]]) -> tp.ContextManager[None]:
    return _scope_context()


@contextmanager
def _scope_context(scope: tp.Optional[tp.Dict[str, tp.Any]]):
    prev_scope = LOCAL.scope
    prev_module_path = LOCAL.module_path

    LOCAL.module_path = {}
    LOCAL.scope = {}

    try:
        yield
    finally:
        LOCAL.scope = prev_scope
        LOCAL.module_path = prev_module_path


def call_context(module: Module) -> tp.ContextManager[None]:
    return _call_context(module)


@contextmanager
def _call_context(module: Module):

    prev_parent = LOCAL.parent
    prev_inside_call = LOCAL.inside_call
    prev_module_index = LOCAL.module_index
    prev_module_path = LOCAL.module_path

    LOCAL.parent = module
    LOCAL.inside_call = True
    LOCAL.module_index = 0

    # this enables one to call a module outside init or apply
    # if LOCAL.module_path is None:
    #     LOCAL.module_path = {}

    assert LOCAL.module_path is not None

    try:
        if prev_parent is not None and module not in prev_parent._submodule_name:
            raise SubmoduleNotRegistered(
                f"Submodule {utils.get_name(module)} not registered in {utils.get_name(prev_parent)}, "
                f"this is probably due to some of the following reasons:\n"
                f"- The submodule is being captured by closure and not registered to any field.\n"
                f"- The submodule was set to a field of the parent but "
                f"its contained inside a more complex type which elegy cannot "
                f"inspect, elegy only looks structures of (possibly nested) list, tuple, or dict.\n"
                f"- The submodule was set to a field of the parent by mutating such field after __init__\n\n"
                f"- If non of the previous is true consider this a bug."
                f"Submodule: {module}\n"
                f"Module: {prev_parent}\n"
            )

        if prev_parent is None:
            LOCAL.module_path[module] = ()
        else:
            parent_path = LOCAL.module_path[prev_parent]
            child_name = prev_parent._submodule_name[module]
            LOCAL.module_path[module] = parent_path + (child_name,)
        yield
    finally:
        LOCAL.parent = prev_parent
        LOCAL.inside_call = prev_inside_call
        LOCAL.module_index = prev_module_index
        LOCAL.module_path = prev_module_path


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
