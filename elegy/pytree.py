import dataclasses
import inspect
import typing as tp
from abc import ABCMeta
from contextlib import contextmanager
from copy import copy

import jax

P = tp.TypeVar("P", bound="PytreeObject")


def field(
    default: tp.Any = dataclasses.MISSING,
    *,
    pytree_node: bool = True,
    default_factory: tp.Any = dataclasses.MISSING,
    init: bool = True,
    repr: bool = True,
    hash: tp.Optional[bool] = None,
    compare: bool = True,
    metadata: tp.Optional[tp.Mapping[str, tp.Any]] = None,
) -> tp.Any:

    if metadata is None:
        metadata = {}
    else:
        metadata = dict(metadata)

    if "pytree_node" in metadata:
        raise ValueError("node is already in metadata")

    metadata["pytree_node"] = pytree_node

    return dataclasses.field(
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
    )


def static_field(
    default: tp.Any = dataclasses.MISSING,
    *,
    default_factory: tp.Any = dataclasses.MISSING,
    init: bool = True,
    repr: bool = True,
    hash: tp.Optional[bool] = None,
    compare: bool = True,
    metadata: tp.Optional[tp.Mapping[str, tp.Any]] = None,
) -> tp.Any:
    return field(
        default=default,
        pytree_node=False,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
    )


class PytreeObjectMeta(ABCMeta):
    def __call__(cls: tp.Type[P], *args: tp.Any, **kwargs: tp.Any) -> tp.Any:

        assert cls._pytree__default_values is not None

        obj = cls.__new__(cls)

        # enable mutability during __init__
        obj.__dict__["_pytree__mutable"] = True
        try:
            # set all '_pytree__*' fields on instances to avoid
            # sharing class variables between instances
            obj._pytree__field_is_node = cls._pytree__field_is_node.copy()
            obj._pytree__default_values = None  # no defaults on instances

            # set default values
            for field, value in cls._pytree__default_values.items():
                setattr(obj, field, value)

            obj = cls.construct(obj, *args, **kwargs)

        finally:
            obj.__dict__["_pytree__mutable"] = False

        for field, value in vars(obj).items():
            if field not in obj._pytree__field_is_node:
                obj._pytree__field_is_node[field] = True

        missing_fields = set(obj._pytree__field_is_node) - set(vars(obj))

        if missing_fields:
            raise TypeError(
                f"Field {missing_fields} where not initialized in {obj.__class__.__name__}"
            )

        return obj

    def construct(cls, obj: P, *args, **kwargs) -> P:
        obj.__init__(*args, **kwargs)
        return obj


class PytreeObject(metaclass=PytreeObjectMeta):
    _pytree__mutable: bool
    _pytree__default_values: tp.Optional[tp.Dict[str, tp.Any]]
    _pytree__field_is_node: tp.Dict[str, bool]

    def __init_subclass__(cls):
        jax.tree_util.register_pytree_node(
            cls,
            flatten_func=tree_flatten,
            unflatten_func=lambda *_args: tree_unflatten(cls, *_args),
        )

        # Restore the signature
        sig = inspect.signature(cls.__init__)
        parameters = tuple(sig.parameters.values())
        cls.__signature__ = sig.replace(parameters=parameters[1:])

        # init class variables
        cls._pytree__mutable = False
        cls._pytree__default_values = {}
        cls._pytree__field_is_node = {}

        # get class info
        class_vars = _get_all_class_vars(cls)
        annotations = _get_all_class_annotations(cls)

        for field, value in class_vars.items():
            if field.startswith("__") or field == "_abc_impl":
                continue
            elif field.startswith("_pytree__"):
                # _pytree__ fields are static
                cls._pytree__field_is_node[field] = False
            elif isinstance(value, dataclasses.Field):
                # save defaults
                if value.default is not dataclasses.MISSING:
                    cls._pytree__default_values[field] = value.default

                # extract metadata
                if value.metadata is not None:
                    cls._pytree__field_is_node[field] = value.metadata.get(
                        "pytree_node", True
                    )
                else:
                    cls._pytree__field_is_node[field] = True
            elif not _is_descriptor(value):
                cls._pytree__field_is_node[field] = True
                cls._pytree__default_values[field] = value

        for field in annotations:
            if field not in cls._pytree__field_is_node:
                cls._pytree__field_is_node[field] = True

                if field in class_vars:
                    cls._pytree__default_values[field] = class_vars[field]

    def replace(self: P, **kwargs: tp.Any) -> P:
        """
        Replace the values of the fields of the object with the values of the
        keyword arguments.
        """
        for key in kwargs:
            if key not in self._pytree__field_is_node:
                raise ValueError(f"'{key}' is not a field of {type(self).__name__}")

        pytree = copy(self)
        pytree.__dict__.update(kwargs)
        return pytree


def tree_flatten(pytree: PytreeObject):

    node_fields = {}
    static_fields = {}

    for field, is_node in pytree._pytree__field_is_node.items():
        value = getattr(pytree, field)
        if is_node:
            node_fields[field] = value
        else:
            static_fields[field] = value

    children = (node_fields,)

    return children, static_fields


def tree_unflatten(cls: tp.Type[P], static_fields, children):

    pytree = cls.__new__(cls)
    (node_fields,) = children

    pytree.__dict__.update(node_fields, **static_fields)

    return pytree


def pytree_setattr(self: P, field: str, value: tp.Any):
    if not self._pytree__mutable:
        raise RuntimeError(
            f"{type(self).__name__} is immutable, trying to update field {field}"
        )

    object.__setattr__(self, field, value)


PytreeObject.__setattr__ = pytree_setattr


@contextmanager
def _make_mutable(pytree: PytreeObject):

    pytree.__dict__["_pytree__mutable"] = True

    try:
        yield
    finally:
        pytree.__dict__["_pytree__mutable"] = False


def _is_descriptor(obj: tp.Any) -> bool:

    for cls in obj.__class__.__mro__:
        if (
            "__get__" in cls.__dict__
            or "__set__" in cls.__dict__
            or "__delete__" in cls.__dict__
        ):
            return True

    return False


def _get_all_class_vars(cls: type) -> tp.Dict[str, tp.Any]:
    d = {}
    for c in reversed(cls.mro()):
        if hasattr(c, "__dict__"):
            d.update(vars(c))
    return d


def _get_all_class_annotations(cls: type) -> tp.Dict[str, type]:
    d = {}
    for c in reversed(cls.mro()):
        if hasattr(c, "__annotations__"):
            d.update(**c.__annotations__)
    return d
