# Implementation based on tf.keras.engine.data_adapter.py
# https://github.com/tensorflow/tensorflow/blob/2b96f3662bd776e277f86997659e61046b56c315/tensorflow/python/keras/engine/data_adapter.py


import abc
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
        # TODO(kaftan): Check performance implications of using a flatten
        #  here for other types of inputs.
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
        **kwargs
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
            for epoch in range(epochs):
                if shuffle:
                    np.random.shuffle(dataset_indices)

                for batch in range(
                    num_full_batches + int(self._partial_batch_size != 0)
                ):
                    indices = dataset_indices[
                        batch * batch_size : (batch + 1) * batch_size
                    ]
                    # Complete missing last batch
                    if len(indices) < batch_size:
                        fill_indices = batch_size - len(indices)
                        indices = np.append(indices, dataset_indices[:fill_indices])

                    data_x = inputs[0][indices]
                    data_y = inputs[1][indices]
                    if len(inputs) == 3:
                        yield (data_x, data_y, inputs[2][indices])
                    else:
                        yield (data_x, data_y)

        self._dataset = dataset_generator()

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

