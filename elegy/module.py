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
    Info,
    InitJit,
    JitCallable,
    MissingParameter,
    NoContext,
    ModuleOrderError,
    Parameter,
    ParameterCollection,
    ParameterSpec,
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


@dataclass
class _LocalContext(threading.local):
    parent: tp.Optional["Module"]
    module_path: tp.Optional[tp.Dict["Module", Path]]
    inside_call: tp.Optional[bool]
    module_index: tp.Optional[int]


LOCAL: LocalContext = _LocalContext(
    parent=None,
    module_path=None,
    inside_call=None,
    module_index=None,
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

                name = utils.get_unique_name(set(parent._submodules), module.name)

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
    _default_params: tp.Dict[str, Parameter]
    _params: tp.Optional[tp.Dict[str, Parameter]]
    _spec: tp.Dict[str, ParameterSpec]
    _initialized: bool = False

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
        self._default_params = {}
        self._params = None
        self._spec = {}
        self._initialized = False

        self._signature_f = self.call

        self._jit_functions()

    @property
    def initialized(self):
        return self._initialized

    def _mark_initialized_recursive(self):
        self._initialized = True
        for submodule in self._submodules.values():
            submodule._mark_initialized_recursive()

    def _register_parameter(self, name: str, parameter: Parameter):
        param_info = jax.tree_map(
            lambda x: Info(shape=x.shape, dtype=x.dtype),
            parameter.value,
        )
        self._spec[name] = ParameterSpec(
            collection=parameter.collection, info=param_info
        )
        self._default_params[name] = parameter

    def defaults_call(self, *args, **kwargs):
        if not self.initialized:
            y, collections = self.init()(*args)
        else:
            collections = self.get_default_parameters()

        y, collections = self.apply(collections)(*args)

        self.set_default_parameters(collections)

        return y

    def defaults_call_jit(self, *args) -> tp.Any:

        if not self.initialized:
            y, collections = self.init_jit()(*args)
        else:
            collections = self.get_default_parameters()

        y, collections = self.apply_jit(collections)(*args)

        self.set_default_parameters(collections)

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
                    self.set_default_parameters(collections)

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
            collections: ParameterCollection,
            *,
            rng: tp.Optional[RNGSeq] = None,
            **hooks_kwargs,
        ) -> JitCallable:
            def apply_jit_callable(
                *args,
            ) -> tp.Tuple[tp.Any, ParameterCollection]:
                with hooks.update_context(rng=rng, **hooks_kwargs):
                    return apply_jit(collections, *args)

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

        # this marks initialization

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
            ), scope_context(self, None):
                y = self(*args, **kwargs)
                collections = self.get_parameters_internal()

            self._mark_initialized_recursive()

            return y, collections

        init_callable._signature_f = self.call

        return init_callable

    def apply(
        self,
        collections: ParameterCollection,
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
            ), scope_context(self, collections):
                y = self(*args, **kwargs)
                collections_ = self.get_parameters_internal()

            return y, collections_

        apply_callable._signature_f = self.call

        return apply_callable

    def __getattr__(self, name: str) -> tp.Any:

        if name in self._submodules:
            return self._submodules[name]

        if self._params is not None and name in self._params:
            return self._params[name].value

        if not has_scope() and self.initialized and name in self._default_params:
            return self._default_params[name].value

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

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

        if self._params is None and hooks.is_initializing():
            self._params = {}

        assert self._params is not None

        if name not in self._params:

            if not hooks.is_initializing():
                raise ValueError(f"Cannot add parameter {name} outside `init`")

            if name in self._default_params:
                parameter = self._default_params[name]
                assert collection == parameter.collection
                initial_value = parameter.value

            else:
                initial_value = initializer()
                initial_value = jax.tree_map(jnp.asarray, initial_value)
                parameter = Parameter(collection, initial_value)
                self._register_parameter(name, parameter)

            self._params[name] = parameter

        parameter = self._params[name]

        if parameter.collection != collection:
            raise ValueError(
                f"Parameter {name} previously found in collection {parameter.collection} "
                f"but currently being added for collection {collection}"
            )

        value = parameter.value

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

        assert self._params is not None

        if name not in self._params:
            raise ValueError(f"Parameter {name} not found in {self}.")

        if hooks.is_initializing():
            return

        parameter = self._params[name]
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
        assert self._params is not None

        if name not in self._params:
            self.add_parameter(
                name, lambda: value, trainable=trainable, collection=collection
            )
        else:
            self.update_parameter(name, value)

    def clear_default_parameters(self):
        self._clear_parameters(default_params=True)

    def _clear_parameters(self, default_params: bool = False):

        self._params = None

        if default_params:
            self._default_params.clear()

        for submodule in self._submodules.values():
            submodule._clear_parameters(default_params=default_params)

    def set_default_parameters(
        self,
        collections: ParameterCollection,
    ):
        self._set_parameters_internal(
            collections=collections,
            set_default_params=True,
        )

    def _set_parameters_internal(
        self,
        collections: ParameterCollection,
        set_default_params: bool = False,
    ):
        old_colletions = self.get_default_parameters()
        collections = jax.tree_map(jnp.asarray, collections)

        try:
            self._set_parameters(
                collections=collections,
                set_default_params=set_default_params,
            )
        except:
            self._set_parameters(
                collections=old_colletions,
                set_default_params=set_default_params,
            )
            raise

    def _validate_parameters(self, collections: ParameterCollection):
        # define sets
        incoming_names = set(
            name for parameters in collections.values() for name in parameters
        )
        expected_submodule_names = set(self._submodules)
        expected_parameter_names = set(self._spec)

        # check missing submodules
        missing_submodules = expected_submodule_names - incoming_names
        if missing_submodules:
            raise ValueError(
                f"Missing submodule parameters {missing_submodules} for module {self.name}"
            )

        # check missing parameters
        missing_parameters = expected_parameter_names - incoming_names
        if missing_parameters:
            raise ValueError(
                f"Missing parameters {missing_parameters} for module {self.name}"
            )

        # check unkown names
        unknown_parameters = (
            incoming_names - expected_submodule_names - expected_parameter_names
        )
        if unknown_parameters:
            raise ValueError(
                f"Got unknown parameters {unknown_parameters} on module {self.name}"
            )

        for name, param_spec in self._spec.items():
            parameter = utils.get_parameter(collections, name)

            if parameter.collection != param_spec.collection:
                raise ValueError(
                    f"Parameter {name} on module {self.name} was expected to be on collection {param_spec.collection} "
                    f"but was found on collection {parameter.collection}"
                )

            def validate_value(value: jnp.ndarray, info: Info):
                if value.shape != info.shape:
                    incoming_shapes = jax.tree_map(lambda x: x.shape, parameter.value)
                    expected_shape = jax.tree_map(lambda x: x.shape, param_spec.info)

                    raise ValueError(
                        f"Shape mismatch in parameter {name} on module {self.name}.\n"
                        f"Got: {incoming_shapes}\n"
                        f"Expected: {expected_shape}"
                    )

            jax.tree_multimap(validate_value, parameter.value, param_spec.info)

    def _set_parameters(
        self,
        collections: ParameterCollection,
        check_shapes: bool = False,
        set_default_params: bool = False,
    ):
        if not self.initialized:
            raise ValueError(f"Cannot set parameters for uninitialized module {self}")
        assert self._spec is not None

        self._validate_parameters(collections)

        # build new parameters
        new_params = {
            name: utils.get_parameter(collections, name) for name in self._spec
        }

        # set module parameters
        if set_default_params:
            self._default_params = new_params
        else:
            self._params = new_params

        # set submodule parameters
        for name, submodule in self._submodules.items():
            submodule._set_parameters(
                collections=utils.get_submodule_colletions(collections, name),
                check_shapes=check_shapes,
                set_default_params=set_default_params,
            )

    def get_default_parameters(self) -> ParameterCollection:
        parameters = self._get_parameters(defaults=True)
        return utils.split_into_collections(parameters)

    def get_parameters_internal(self, defaults: bool = False) -> ParameterCollection:
        parameters = self._get_parameters(defaults=defaults)
        return utils.split_into_collections(parameters)

    def _get_parameters(self, defaults: bool) -> tp.Dict[str, tp.Any]:
        if (self._params is None and not defaults) or (
            not self.initialized and defaults
        ):
            return {}

        parameters: tp.Dict[str, tp.Any]

        if defaults:
            assert self.initialized
            parameters = self._default_params.copy()
        else:
            assert self._params is not None
            parameters = self._params.copy()

        for name, submodule in self._submodules.items():
            parameters[name] = submodule._get_parameters(defaults=defaults)

        return parameters

    def has_parameter(self, name: str) -> bool:
        return hasattr(self, name)


# -------------------------------------------------------------
# hooks
# -------------------------------------------------------------


def get_module_path(module: tp.Optional[Module] = None) -> tp.Optional[Path]:
    if module is None:
        module = LOCAL.parent

    return LOCAL.module_path[module] if module is not None and has_scope() else None


def get_module_path_str(module: Module) -> tp.Optional[str]:
    return (
        "/".join(map(str, LOCAL.module_path[module]))
        if module is not None and has_scope()
        else None
    )


def has_scope() -> bool:
    return LOCAL.module_path is not None


# -----------------------------------------------------------------------------
# context managers
# -----------------------------------------------------------------------------


def scope_context(
    module: Module, collections: tp.Optional[ParameterCollection]
) -> tp.ContextManager[None]:
    return _scope_context(module, collections)


@contextmanager
def _scope_context(module: Module, collections: tp.Optional[ParameterCollection]):
    prev_module_path = LOCAL.module_path

    LOCAL.module_path = {}

    try:
        if collections is not None:
            module._set_parameters_internal(collections)
        yield
    finally:
        module._clear_parameters()
        LOCAL.module_path = prev_module_path


def call_context(module: Module) -> tp.ContextManager[None]:
    return _call_context(module)


@contextmanager
def _call_context(module: Module):

    prev_parent = LOCAL.parent
    prev_inside_call = LOCAL.inside_call
    prev_module_index = LOCAL.module_index

    LOCAL.parent = module
    LOCAL.inside_call = True
    LOCAL.module_index = 0

    try:

        if not has_scope():
            raise NoContext(
                f"Trying to call top-level module '{module}' directly, use `apply` instead."
            )

        elif prev_parent is not None and module not in prev_parent._submodule_name:
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


def instantiation_context(module: Module) -> tp.ContextManager[None]:
    return _instantiation_context(module)


@contextmanager
def _instantiation_context(module: Module):

    prev_module = LOCAL.parent
    prev_inside_call = LOCAL.inside_call
    prev_module_path = LOCAL.module_path

    LOCAL.inside_call = False
    LOCAL.module_path = None
    LOCAL.parent = module

    try:
        yield
    finally:
        LOCAL.parent = prev_module
        LOCAL.inside_call = prev_inside_call
        LOCAL.module_path = prev_module_path


# ------------------------------------------------------------------------
# utils
# ------------------------------------------------------------------------


def leaf_isinstance(obj: tp.Any, types) -> bool:

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
            self._signature_f = f
            self.f = f

        def call(self, *args, **kwargs):
            return self.f(*args, **kwargs)

    ToModule.__name__ = f.__name__

    return ToModule
