# Implementation based on tf.keras.engine.data_adapter.py
# https://github.com/tensorflow/tensorflow/blob/2b96f3662bd776e277f86997659e61046b56c315/tensorflow/python/keras/engine/data_adapter.py


import collections
import functools
import math
import typing as tp

import jax.numpy as jnp
import numpy as np
import six
from elegy import types, utils


class Multimap(types.Protocol):
    def __call__(self, *args: types.np.ndarray) -> types.T:
        ...


def map_append(output, batch_output):
    output.append(batch_output)
    return output


def map_structure(
    f: Multimap,
    *structure: tp.Union[types.ArrayHolder, None],
) -> types.Container[tp.Union[types.T, None]]:

    if isinstance(structure[0], tp.Tuple):
        return tuple(map_structure(f, *x) for x in zip(*structure))
    elif isinstance(structure[0], tp.Dict):
        return {
            key: map_structure(f, *(elem[key] for elem in structure))
            for key in structure[0]
        }
    elif structure[0] is not None:
        return f(*structure)
    else:  # if it is tuple of None
        return None


def flatten(inputs: types.ArrayHolder) -> tp.Iterable[types.np.ndarray]:

    if isinstance(inputs, (jnp.ndarray, np.ndarray)):
        yield inputs
    elif isinstance(inputs, tp.Tuple):
        for x in inputs:
            yield from flatten(x)
    elif isinstance(inputs, tp.Dict):
        for x in inputs.values():
            yield from flatten(x)
    elif isinstance(inputs, (tp.Generator, tp.Iterator, tp.Iterable)):
        yield inputs
    else:
        raise TypeError(f"Unsupported type '{type(inputs)}'")


def pack_x_y_sample_weight(x, y=None, sample_weight=None):
    """Packs user-provided data into a tuple."""
    if y is None:
        return (x,)
    elif sample_weight is None:
        return (x, y)
    else:
        return (x, y, sample_weight)


def unpack_x_y_sample_weight(data):
    """Unpacks user-provided data tuple."""
    if not isinstance(data, tuple):
        return (data, None, None)
    elif len(data) == 1:
        return (data[0], None, None)
    elif len(data) == 2:
        return (data[0], data[1], None)
    elif len(data) == 3:
        return (data[0], data[1], data[2])

    raise ValueError("Data not understood.")


def list_to_tuple(maybe_list):
    if isinstance(maybe_list, list):
        return tuple(maybe_list)
    return maybe_list


def handle_partial_sample_weights(y, sample_weights):
    any_sample_weight = sample_weights is not None and any(
        w is not None for w in sample_weights
    )
    partial_sample_weight = any_sample_weight and any(w is None for w in sample_weights)

    if not any_sample_weight:
        return None

    if not partial_sample_weight:
        return sample_weights


def is_none_or_empty(inputs):
    # util method to check if the input is a None or a empty list.
    # the python "not" check will raise an error like below if the input is a
    # numpy array
    # "The truth value of an array with more than one element is ambiguous.
    # Use a.any() or a.all()"
    return inputs is None or not list(flatten(inputs))


def assert_not_namedtuple(x):
    if (
        isinstance(x, tuple)
        and
        # TODO(b/144192902): Use a namedtuple checking utility.
        hasattr(x, "_fields")
        and isinstance(x._fields, collections.Sequence)
        and all(isinstance(f, six.string_types) for f in x._fields)
    ):
        raise ValueError(
            "Received namedtuple ({}) with fields `{}` as input. namedtuples "
            "cannot, in general, be unambiguously resolved into `x`, `y`, "
            "and `sample_weight`. For this reason Keras has elected not to "
            "support them. If you would like the value to be unpacked, "
            "please explicitly convert it to a tuple before passing it to "
            "Keras.".format(x.__class__, x._fields)
        )


def train_validation_split(arrays, validation_split, shuffle=True) -> tp.Tuple:
    """
    Split arrays into random train and validation subsets.
    Arguments:
        arrays: Arrays to split. Allowed inputs are arbitrarily nested structures
            of Jax and NumPy arrays.
        validation_split: Float between 0 and 1. The proportion of the dataset to
            include in the validation split. The rest of the dataset will be included
            in the training split.
        shuffle: Bool. Whether to shuffle the data before performing a split. If
            `False`, the last `validation_split` fraction of that training data will
            become the validation split.
    Returns:
        `(train_arrays, validation_arrays)`
    """

    def _can_split(t):
        supported_types = (jnp.ndarray, np.ndarray)
        # if pd:
        #     supported_types = (jnp.ndarray, np.ndarray, pd.Series, pd.DataFrame)
        return isinstance(t, supported_types) or t is None

    flat_arrays = flatten(arrays)
    # flat_arrays = arrays
    if not all(_can_split(t) for t in arrays):
        raise ValueError(
            "`validation_split` is only supported for Tensors or NumPy "
            "arrays, found: {}".format(arrays)
        )

    if all(t is None for t in flat_arrays):
        return arrays, arrays

    first_non_none = None
    for t in flat_arrays:
        if t is not None:
            first_non_none = t
            break

    # Assumes all arrays have the same batch shape or are `None`.
    batch_dim = int(first_non_none.shape[0])
    indices = np.arange(batch_dim)
    if shuffle:
        indices = np.random.shuffle(indices)
    split_at = int(math.floor(batch_dim * (1.0 - validation_split)))
    train_indices = indices[:split_at]
    val_indices = indices[split_at:]

    def _split(t, indices):
        if t is None:
            return t
        return t[indices]

    train_arrays = map_structure(
        functools.partial(_split, indices=train_indices), arrays
    )
    val_arrays = map_structure(functools.partial(_split, indices=val_indices), arrays)
    return train_arrays, val_arrays
