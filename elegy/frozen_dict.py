# taken from https://github.com/google/flax/blob/cb90727856903e9aaddd8aa5f893a7ff0cbfa2d5/flax/core/frozen_dict.py

"""Frozen Dictionary."""

import typing as tp
from typing import Dict, Mapping, TypeVar

import jax
import jax.tree_util

K = TypeVar("K")
V = TypeVar("V")


@jax.tree_util.register_pytree_node_class
class FrozenDict(Mapping[K, V]):
    """An immutable variant of the Python dict."""

    __slots__ = ("_dict", "_hash")

    def __init__(self, *args, **kwargs):
        self._dict = dict(*args, **kwargs)
        self._hash = None

    def __getitem__(self, key):
        v = self._dict[key]
        if isinstance(v, dict):
            return FrozenDict(v)
        return v

    def __setitem__(self, key, value):
        raise ValueError("FrozenDict is immutable.")

    def __contains__(self, key):
        return key in self._dict

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __repr__(self):
        return "FrozenDict(%r)" % unfreeze(self._dict)

    def __hash__(self):
        if self._hash is None:
            h = 0
            for key, value in self.items():
                h ^= hash((key, value))
            self._hash = h
        return self._hash

    def copy(self, **add_or_replace):
        return type(self)(self, **add_or_replace)

    def items(self):
        for key in self._dict:
            yield (key, self[key])

    def unfreeze(self) -> Dict[K, V]:
        return unfreeze(self)

    def tree_flatten(self):
        return (self._dict,), ()

    @classmethod
    def tree_unflatten(cls, _, data):
        return cls(*data)


def freeze(xs: tp.Union[tp.Dict[K, V], FrozenDict[K, V]]) -> FrozenDict[K, V]:
    """Freeze a nested dict.
    Makes a nested `dict` immutable by transforming it into `FrozenDict`.
    """
    # Turn the nested FrozenDict into a dict. This way the internal data structure
    # of FrozenDict does not contain any FrozenDicts.
    # instead we create those lazily in `__getitem__`.
    # As a result tree_flatten/unflatten will be fast
    # because it operates on native dicts.
    xs = unfreeze(xs)
    return FrozenDict(xs)


def unfreeze(x: tp.Union[tp.Dict[K, V], FrozenDict[K, V], tp.Any]) -> Dict[K, V]:
    """Unfreeze a FrozenDict.
    Makes a mutable copy of a `FrozenDict` mutable by transforming
    it into (nested) dict.
    """
    if not isinstance(x, (FrozenDict, tp.Dict)):
        return x

    ys = {}
    for key, value in x.items():
        ys[key] = unfreeze(value)
    return ys
