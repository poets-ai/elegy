# Implementation based on tf.keras.engine.data_adapter.py
# https://github.com/tensorflow/tensorflow/blob/2b96f3662bd776e277f86997659e61046b56c315/tensorflow/python/keras/engine/data_adapter.py


from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.data.ops import dataset_ops

from .data_adapter import DataAdapter
from .utils import is_none_or_empty, map_structure, flatten


class TFDatasetAdapter(DataAdapter):
    """Adapter that handles `tf.data.Dataset`."""

    @staticmethod
    def can_handle(x, y=None):
        return isinstance(x, (dataset_ops.DatasetV1, dataset_ops.DatasetV2))

    def __init__(self, x, y=None, sample_weights=None, steps=None, **kwargs):
        super().__init__(x, y, **kwargs)
        # Note that the dataset instance is immutable, its fine to reuse the user
        # provided dataset.
        self._dataset = x

        # The user-provided steps.
        self._user_steps = steps

        self._validate_args(y, sample_weights, steps)

        # Since we have to know the dtype of the dataset when we build the
        # dataset, we have to look at a batch to infer the structure.
        peek = next(iter(x))

        self._first_batch_size = int(list(flatten(peek))[0].shape[0])

    def get_dataset(self):
        def parse_tf_data_gen():
            for batch in iter(self._dataset):
                batch = map_structure(lambda x: x.numpy(), batch)
                yield batch

        return parse_tf_data_gen

    def get_size(self):
        size = cardinality.cardinality(self._dataset)
        if size == cardinality.INFINITE and self._user_steps is None:
            raise ValueError(
                "When passing an infinitely repeating tf.data.Dataset, you "
                "must specify how many steps to draw."
            )
        elif size == cardinality.INFINITE:
            return self._user_steps
        elif size >= 0:
            return size.numpy().item()

    @property
    def batch_size(self):
        return self.representative_batch_size

    @property
    def representative_batch_size(self):
        return self._first_batch_size

    @property
    def partial_batch_size(self):
        return

    def has_partial_batch(self):
        return False

    def should_recreate_iterator(self):
        # If user doesn't supply `steps`, or if they supply `steps` that
        # exactly equals the size of the `Dataset`, create a new iterator
        # each epoch.
        return (
            self._user_steps is None
            or cardinality.cardinality(self._dataset).numpy() == self._user_steps
        )

    def _validate_args(self, y, sample_weights, steps):
        """Validates `__init__` arguments."""
        # Arguments that shouldn't be passed.
        if not is_none_or_empty(y):
            raise ValueError(
                "`y` argument is not supported when using " "tf.Data.dataset as input."
            )
        if not is_none_or_empty(sample_weights):
            raise ValueError(
                "`sample_weight` argument is not supported when using "
                "tf.Data.dataset as input."
            )

        size = cardinality.cardinality(self._dataset).numpy()
        if size == cardinality.INFINITE and steps is None:
            raise ValueError(
                "When providing an infinitely repeating tf.data.Dataset, you must specify "
                "the number of steps to run."
            )
