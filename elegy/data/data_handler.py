# Implementation based on tf.keras.engine.data_adapter.py
# https://github.com/tensorflow/tensorflow/blob/2b96f3662bd776e277f86997659e61046b56c315/tensorflow/python/keras/engine/data_adapter.py


import contextlib
import logging

from .array_adapter import ArrayDataAdapter
from .generator_adapter import GeneratorDataAdapter
from .list_adapter import ListsOfScalarsDataAdapter
from .dataset import DataLoaderAdapter

try:
    from .tf_dataset_adapter import TFDatasetAdapter
except ImportError:
    TFDatasetAdapter = None
try:
    from .torch_dataloader_adapter import TorchDataLoaderAdapter
except ImportError:
    TorchDataLoaderAdapter = None

ALL_ADAPTER_CLS = [
    ArrayDataAdapter,
    GeneratorDataAdapter,
    ListsOfScalarsDataAdapter,
    DataLoaderAdapter,
]

if TFDatasetAdapter is not None:
    ALL_ADAPTER_CLS.append(TFDatasetAdapter)
if TorchDataLoaderAdapter is not None:
    ALL_ADAPTER_CLS.append(TorchDataLoaderAdapter)


class DataHandler(object):
    """Handles iterating over epoch-level `tp.Iterator` objects."""

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
        """Yields `(epoch, tp.Iterator)`."""
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
                    "if using tf.data.Dataset.".format(
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
        (1) A generator `Dataset` was passed to the `DataHandler`, and
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
            "When passing a generator, you " "must specify how many steps to draw."
        )

    @property
    def _samples(self):
        return self._adapter.get_samples()

    @property
    def batch_size(self):
        return self._adapter.batch_size


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
