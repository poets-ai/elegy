# Implementation based on tf.keras.engine.data_adapter.py
# https://github.com/tensorflow/tensorflow/blob/2b96f3662bd776e277f86997659e61046b56c315/tensorflow/python/keras/engine/data_adapter.py


import abc
import collections
import contextlib
import itertools
import logging
import math
import typing as tp

import jax.numpy as jnp
import numpy as np
import six

# from tensorflow.python.keras.engine import training_utils
# from tensorflow.python.util import nest


@six.add_metaclass(abc.ABCMeta)
class DataAdapter(object):
    """Base class for input data adapter.
  In TF 2.0, tf.data is the preferred API for user to feed in data. In order
  to simplify the training code path, all the input data object will be
  converted to `tf.data.Dataset` if possible.
  Note that since this class is mainly targeted for TF 2.0, it might have a lot
  of assumptions under the hood, eg eager context by default, distribution
  strategy, etc. In the meantime, some legacy feature support might be dropped,
  eg, Iterator from dataset API in v1, etc.
  The sample usage of this class is like:
  ```
  x = tf.data.Dataset.range(100)
  adapter_cls = [NumpyArrayDataAdapter, ..., DatasetAdapter]
  applicable_adapters = [cls for cls in adapter_cls if cls.can_handle(x)]
  if len(applicable_adapters) != 1:
    raise ValueError("Expect only one adapter class to handle the input")
  dataset = applicable_adapters[0](x).get_dataset()
  for data in dataset:
    # training
  ```
  """

    @staticmethod
    def can_handle(x, y=None):
        """Whether the current DataAdapter could handle the input x and y.
    Structure wise, x and y can be single object, or list of objects if there
    multiple input/output, or dictionary of objects when the intput/output are
    named.
    Args:
      x: input features.
      y: target labels. Note that y could be None in the case of prediction.
    Returns:
      boolean
    """
        raise NotImplementedError

    @abc.abstractmethod
    def __init__(self, x, y=None, **kwargs):
        """Create a DataAdapter based on data inputs.
    The caller must make sure to call `can_handle()` first before invoking this
    method. Provide unsupported data type will result into unexpected behavior.
    Args:
      x: input features.
      y: target labels. Note that y could be None in the case of prediction.
      **kwargs: Other keyword arguments for DataAdapter during the construction
        of the tf.dataset.Dataset. For example:
        - Numpy data might have `sample_weights` which will be used for
          weighting the loss function during training.
        - Numpy data might need to have `batch_size` parameter when constructing
          the dataset and iterator.
        - Certain input might need to be distribution strategy aware. When
          `distribution_strategy` is passed, the created dataset need to respect
          the strategy.
        DataAdapter might choose to ignore any keyword argument if it doesn't
        use it, or raise exception if any required argument is not provide.
    """
        if not self.can_handle(x, y):
            raise ValueError(
                "{} Cannot handle input {}, {}".format(self.__class__, x, y)
            )

    @abc.abstractmethod
    def get_dataset(self):
        """Get a dataset instance for the current DataAdapter.
    Note that the dataset returned does not repeat for epoch, so caller might
    need to create new iterator for the same dataset at the beginning of the
    epoch. This behavior might change in future.
    Returns:
      An tf.dataset.Dataset. Caller might use the dataset in different
      context, eg iter(dataset) in eager to get the value directly, or in graph
      mode, provide the iterator tensor to Keras model function.
    """
        raise NotImplementedError

    @abc.abstractmethod
    def get_size(self):
        """Return the size (number of batches) for the dataset created.
    For certain type of the data input, the number of batches is known, eg for
    Numpy data, the size is same as (number_of_element / batch_size). Whereas
    for dataset or python generator, the size is unknown since it may or may not
    have a end state.
    Returns:
      int, the number of batches for the dataset, or None if it is unknown. The
      caller could use this to control the loop of training, show progress bar,
      or handle unexpected StopIteration error.
    """
        raise NotImplementedError

    @abc.abstractmethod
    def batch_size(self):
        """Return the batch size of the dataset created.
    For certain type of the data input, the batch size is known, and even
    required, like numpy array. Where as for dataset, the batch is unknown
    unless we take a peek.
    Returns:
      int, the batch size of the dataset, or None if it is unknown.
    """
        raise NotImplementedError

    def representative_batch_size(self):
        """Return a representative size for batches in the dataset.
    This is not guaranteed to be the batch size for all batches in the
    dataset. It just needs to be a rough approximation for batch sizes in
    the dataset.
    Returns:
      int, a representative size for batches found in the dataset,
      or None if it is unknown.
    """
        return self.batch_size()

    @abc.abstractmethod
    def has_partial_batch(self):
        """Whether the dataset has partial batch at the end."""
        raise NotImplementedError

    @abc.abstractmethod
    def partial_batch_size(self):
        """The size of the final partial batch for dataset.
    Will return None if has_partial_batch is False or batch_size is None.
    """
        raise NotImplementedError

    @abc.abstractmethod
    def should_recreate_iterator(self):
        """Returns whether a new iterator should be created every epoch."""
        raise NotImplementedError

    def get_samples(self):
        """Returns number of samples in the data, or `None`."""
        if not self.get_size() or not self.batch_size():
            return None
        total_sample = self.get_size() * self.batch_size()
        if self.has_partial_batch():
            total_sample -= self.batch_size() - self.partial_batch_size()
        return total_sample

    def on_epoch_end(self):
        """A hook called after each epoch."""
        pass


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


class ArrayDataAdapter(DataAdapter):
    """Adapter that handles NumPy and Jax numpy arrays."""

    @staticmethod
    def can_handle(x, y=None):
        data = [x]
        if y is not None:
            data += [y]

        supported_types = (jnp.ndarray, np.ndarray)
        # if pd:
        #     supported_types = (ops.Tensor, np.ndarray, pd.Series, pd.DataFrame)

        def _is_array(v):
            if isinstance(v, supported_types):
                return True
            return False

        return all(_is_array(v) for v in data)

    def __init__(
        self,
        x: tp.Union[jnp.ndarray, np.ndarray],
        y: tp.Union[jnp.ndarray, np.ndarray, None] = None,
        sample_weights: tp.Union[jnp.ndarray, np.ndarray, None] = None,
        batch_size: tp.Optional[int] = None,
        epochs: int = 1,
        steps: tp.Optional[int] = None,
        shuffle: bool = False,
        drop_remainder: bool = False,
        **kwargs,
    ):
        super(ArrayDataAdapter, self).__init__(x, y, **kwargs)
        # x, y, sample_weights = _process_tensorlike((x, y, sample_weights))
        # sample_weight_modes = broadcast_sample_weight_modes(
        #     sample_weights, sample_weight_modes
        # )

        # If sample_weights are not specified for an output use 1.0 as weights.
        # (sample_weights, _, _) = training_utils.handle_partial_sample_weights(
        #     y, sample_weights, sample_weight_modes, check_all_flat=True
        # )
        # sample_weights = handle_partial_sample_weights(y, sample_weights)

        inputs = pack_x_y_sample_weight(x, y, sample_weights)

        # num_samples = set(int(i.shape[0]) for i in nest.flatten(inputs))
        num_samples = set(int(i.shape[0]) for i in inputs)
        if len(num_samples) > 1:
            msg = "Data cardinality is ambiguous:\n"
            for label, data in zip(["x", "y", "sample_weight"], inputs):
                msg += "  {} sizes: {}\n".format(
                    label, ", ".join(str(i.shape[0]) for i in data)
                )
            msg += "Please provide data which shares the same first dimension."
            raise ValueError(msg)
        num_samples = num_samples.pop()

        # If batch_size is not passed but steps is, calculate from the input data.
        if not batch_size:
            batch_size = int(math.ceil(num_samples / steps)) if steps else None
            if batch_size is None:
                raise ValueError("Please provide either batch_size or steps")

        self._size = int(math.ceil(num_samples / batch_size))
        self._batch_size = batch_size

        num_full_batches = int(num_samples // batch_size)
        self._partial_batch_size = num_samples % batch_size

        self._shuffle = shuffle

        dataset_indices = np.arange(num_samples)

        def dataset_generator():
            while True:
                if shuffle:
                    np.random.shuffle(dataset_indices)

                for batch in range(
                    num_full_batches + int(self._partial_batch_size != 0)
                ):
                    indices = dataset_indices[
                        batch * batch_size : (batch + 1) * batch_size
                    ]

                    # # Drop last batch
                    # if drop_remainder and len(indices) < batch_size:
                    #     print("Droping!")
                    #     continue

                    data_x = inputs[0][indices]
                    data_y = inputs[1][indices]
                    if len(inputs) == 3:
                        yield (data_x, data_y, inputs[2][indices])
                    else:
                        yield (data_x, data_y)

        self._dataset = dataset_generator

    def get_dataset(self):
        return self._dataset

    def get_size(self):
        return self._size

    def batch_size(self):
        return self._batch_size

    def has_partial_batch(self):
        return self._partial_batch_size > 0

    def partial_batch_size(self):
        return self._partial_batch_size or None

    def should_recreate_iterator(self):
        # An infinite dataset is always created here.
        return False


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


class GeneratorDataAdapter(DataAdapter):
    """Adapter that handles python generators and iterators."""

    @staticmethod
    def can_handle(x, y=None):
        return (hasattr(x, "__next__") or hasattr(x, "next")) and hasattr(x, "__iter__")

    def __init__(
        self, x: tp.Union[tp.Iterable], y=None, sample_weights=None, **kwargs,
    ):
        # Generators should never shuffle as exhausting the generator in order to
        # shuffle the batches is inefficient.
        kwargs.pop("shuffle", None)

        if not is_none_or_empty(y):
            raise ValueError(
                "`y` argument is not supported when using " "python generator as input."
            )
        if not is_none_or_empty(sample_weights):
            raise ValueError(
                "`sample_weight` argument is not supported when using "
                "python generator as input."
            )

        super(GeneratorDataAdapter, self).__init__(x, y, **kwargs)

        # Since we have to know the dtype of the python generator when we build the
        # dataset, we have to look at a batch to infer the structure.
        peek, x = self._peek_and_restore(x)
        assert_not_namedtuple(peek)
        peek = self._standardize_batch(peek)
        # peek = _process_tensorlike(peek)

        # self._first_batch_size = int(nest.flatten(peek)[0].shape[0])
        self._first_batch_size = int(peek[0].shape[0])

        # Note that dataset API takes a callable that creates a generator object,
        # rather than generator itself, which is why we define a function here.
        generator_fn = lambda: x

        def wrapped_generator():
            for data in generator_fn():
                yield self._standardize_batch(data)

        dataset = generator_fn

        self._dataset = dataset

    def _standardize_batch(self, data):
        """Standardizes a batch output by a generator."""
        # Removes `None`s.
        x, y, sample_weight = unpack_x_y_sample_weight(data)
        data = pack_x_y_sample_weight(x, y, sample_weight)

        # data = nest._list_to_tuple(data)  # pylint: disable=protected-access

        # def _convert_dtype(t):
        #     if isinstance(t, np.ndarray) and issubclass(t.dtype.type, np.floating):
        #         return np.array(t, dtype=backend.floatx())
        #     return t

        # data = nest.map_structure(_convert_dtype, data)
        return data

    @staticmethod
    def _peek_and_restore(x):
        peek = next(x)
        return peek, itertools.chain([peek], x)

    def get_dataset(self):
        return self._dataset

    def get_size(self):
        return None

    def batch_size(self):
        return None

    def representative_batch_size(self):
        return self._first_batch_size

    def has_partial_batch(self):
        return False

    def partial_batch_size(self):
        return

    def should_recreate_iterator(self):
        return False


ALL_ADAPTER_CLS = [ArrayDataAdapter, GeneratorDataAdapter]
# ALL_ADAPTER_CLS = [
#     ListsOfScalarsDataAdapter,
#     TensorLikeDataAdapter,
#     GenericArrayLikeDataAdapter,
#     DatasetAdapter,
#     GeneratorDataAdapter,
#     KerasSequenceAdapter,
#     CompositeTensorDataAdapter,
# ]


def _type_name(x):
    """Generates a description of the type of an object."""
    if isinstance(x, dict):
        key_types = set(_type_name(key) for key in x.keys())
        val_types = set(_type_name(key) for key in x.values())
        return "({} containing {} keys and {} values)".format(
            type(x), key_types, val_types
        )
    if isinstance(x, (list, tuple)):
        types = set(_type_name(val) for val in x)
        return "({} containing values of types {})".format(type(x), types)
    return str(type(x))


def select_data_adapter(x, y):
    """Selects a data adapter than can handle a given x and y."""
    adapter_cls = [cls for cls in ALL_ADAPTER_CLS if cls.can_handle(x, y)]
    if not adapter_cls:
        raise ValueError(
            "Failed to find data adapter that can handle "
            "input: {}, {}".format(_type_name(x), _type_name(y))
        )
    elif len(adapter_cls) > 1:
        raise RuntimeError(
            "Data adapters should be mutually exclusive for "
            "handling inputs. Found multiple adapters {} to handle "
            "input: {}, {}".format(adapter_cls, _type_name(x), _type_name(y))
        )
    return adapter_cls[0]


# from tensorflow.python.eager import context
# from tensorflow.python.framework import errors


class DataHandler(object):
    """Handles iterating over epoch-level `tf.data.Iterator` objects."""

    def __init__(
        self,
        x,
        y=None,
        sample_weight=None,
        batch_size=None,
        steps_per_epoch=None,
        initial_epoch=0,
        epochs=1,
        shuffle=False,
        class_weight=None,
        **kwargs,
    ):

        self._initial_epoch = initial_epoch
        self._epochs = epochs
        self._insufficient_data = False

        adapter_cls = select_data_adapter(x, y)
        self._adapter = adapter_cls(
            x,
            y,
            batch_size=batch_size,
            steps=steps_per_epoch,
            epochs=epochs - initial_epoch,
            sample_weights=sample_weight,
            shuffle=shuffle,
            **kwargs,
        )

        dataset = self._adapter.get_dataset()

        self._inferred_steps = self._infer_steps(steps_per_epoch, dataset)
        self._dataset = dataset

    def enumerate_epochs(self):
        """Yields `(epoch, tf.data.Iterator)`."""
        data_iterator = self._dataset()
        for epoch in range(self._initial_epoch, self._epochs):
            if self._insufficient_data:  # Set by `catch_stop_iteration`.
                break
            if self._adapter.should_recreate_iterator():
                data_iterator = self._dataset()
            yield epoch, data_iterator
            self._adapter.on_epoch_end()

    @contextlib.contextmanager
    def catch_stop_iteration(self):
        """Catches errors when an iterator runs out of data."""
        try:
            yield
            # context.async_wait()
        except (StopIteration):
            if (
                self._adapter.get_size() is None
                and self._inferred_steps is None
                and self._current_step > 0
            ):
                # The input passed by the user ran out of batches.
                # Now we know the cardinality of the input(dataset or generator).
                self._inferred_steps = self._current_step
            else:
                self._insufficient_data = True
                total_epochs = self._epochs - self._initial_epoch
                logging.warning(
                    "Your input ran out of data; interrupting training. "
                    "Make sure that your dataset or generator can generate at "
                    "least `steps_per_epoch * epochs` batches (in this case, "
                    "{} batches). You may need to use the repeat() function "
                    "when building your dataset.".format(
                        total_epochs * self._inferred_steps
                    )
                )

    def steps(self):
        """Yields steps for the current epoch."""
        self._current_step = 0
        # `self._inferred_steps` can be changed by `catch_stop_iteration`.
        while self._inferred_steps is None or self._current_step < self._inferred_steps:
            if self._insufficient_data:  # Set by `catch_stop_iteration`.
                break
            yield self._current_step
            self._current_step += 1

    @property
    def inferred_steps(self):
        """The inferred steps per epoch of the created `Dataset`.
            This will be `None` in the case where:
            (1) A `Dataset` of unknown cardinality was passed to the `DataHandler`, and
            (2) `steps_per_epoch` was not provided, and
            (3) The first epoch of iteration has not yet completed.
            Returns:
            The inferred steps per epoch of the created `Dataset`.
        """
        return self._inferred_steps

    def _infer_steps(self, steps, dataset):
        """Infers steps_per_epoch needed to loop through a dataset."""
        if steps is not None:
            return steps

        adapter_steps = self._adapter.get_size()
        if adapter_steps is not None:
            return adapter_steps

        raise ValueError(
            "When passing an infinitely repeating dataset, you "
            "must specify how many steps to draw."
        )
        # size = cardinality.cardinality(dataset)
        # if size == cardinality.INFINITE and steps is None:
        #     raise ValueError(
        #         "When passing an infinitely repeating dataset, you "
        #         "must specify how many steps to draw."
        #     )
        # if size >= 0:
        #     return size.numpy().item()
        # return None

    @property
    def _samples(self):
        return self._adapter.get_samples()

