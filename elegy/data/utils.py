import collections

import six


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
    """Datasets will stack the list of tensor, so switch them to tuples."""
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
    # return inputs is None or not nest.flatten(inputs)
    return inputs is None or not inputs


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
