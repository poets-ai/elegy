# Implementation based on elegy.engine.data_adapter.py
# https://github.com/tensorflow/tensorflow/blob/2b96f3662bd776e277f86997659e61046b56c315/tensorflow/python/keras/engine/data_adapter.py


import abc
import typing as tp

import six


@six.add_metaclass(abc.ABCMeta)
class DataAdapter(object):
    """Base class for input data adapter.
    In order to simplify the training code path, all the input data
    object will be converted to a `generator` if possible.
    The sample usage of this class is like:

    ```
    x = list(range(100))
    adapter_cls = [ArrayDataAdapter, ..., ListsOfScalarsDataAdapter]
    applicable_adapters = [cls for cls in adapter_cls if cls.can_handle(x)]
    if len(applicable_adapters) != 1:
      raise ValueError("Expect only one adapter class to handle the input")
    dataset = applicable_adapters[0](x).get_dataset()
    for data in dataset():
      # training
    ```
    """

    @staticmethod
    def can_handle(x, y=None):
        """Whether the current DataAdapter could handle the input x and y.
        Structure wise, x and y can be single object, or list of objects if there
        multiple input/output, or dictionary of objects when the input/output are
        named.
        Arguments:
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

        Arguments:
          x: input features.
          y: target labels. Note that y could be None in the case of prediction.
          **kwargs: Other keyword arguments for DataAdapter during the construction
            of the generator. For example:

              - Numpy data might have `sample_weights` which will be used for
                weighting the loss function during training.
              - Numpy data might need to have `batch_size` parameter when constructing
                the dataset and iterator.

            DataAdapter might choose to ignore any keyword argument if it doesn't
            use it, or raise exception if any required argument is not provide.
        """
        if not self.can_handle(x, y):
            raise ValueError(
                "{} Cannot handle input {}, {}".format(self.__class__, x, y)
            )

    @abc.abstractmethod
    def get_dataset(self):
        """Get a function that returns a generator for the current DataAdapter.
        Note that the generator wrapped in the function will repeat for each epoch,
        so the steps for traversing it should be known.
        Returns:
          An function wrapping a generator.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_size(self):
        """Return the size (number of batches) for the dataset created.
        For certain type of the data input, the number of batches is known, eg for
        Numpy data, the size is same as (number_of_element / batch_size). Whereas
        for python generator, the size is unknown since it may or may not
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
        required, like numpy array. Where as for generator, the batch is unknown
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
