# Implementation based on tf.keras.engine.data_adapter.py
# https://github.com/tensorflow/tensorflow/blob/2b96f3662bd776e277f86997659e61046b56c315/tensorflow/python/keras/engine/data_adapter.py


import typing as tp

import numpy as np

from .array_adapter import ArrayDataAdapter
from .data_adapter import DataAdapter

scalar_types = (float, int, str)


class ListsOfScalarsDataAdapter(DataAdapter):
    """Adapter that handles lists of scalars and lists of lists of scalars."""

    @staticmethod
    def can_handle(x, y=None):
        handles_x = ListsOfScalarsDataAdapter._is_list_of_scalars(x)
        handles_y = True
        if y is not None:
            handles_y = ListsOfScalarsDataAdapter._is_list_of_scalars(y)
        return handles_x and handles_y

    @staticmethod
    def _is_list_of_scalars(inp):
        if isinstance(inp, scalar_types):
            return True
        if isinstance(inp, (list, tuple)):
            return ListsOfScalarsDataAdapter._is_list_of_scalars(inp[0])
        return False

    def __init__(
        self, x, y=None, sample_weights=None, batch_size=None, shuffle=False, **kwargs
    ):
        super(ListsOfScalarsDataAdapter, self).__init__(x, y, **kwargs)
        x = np.asarray(x)
        if y is not None:
            y = np.asarray(y)
        if sample_weights is not None:
            sample_weights = np.asarray(sample_weights)

        self._internal_adapter = ArrayDataAdapter(
            x,
            y=y,
            sample_weights=sample_weights,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs
        )

    def get_dataset(self):
        return self._internal_adapter.get_dataset()

    def get_size(self):
        return self._internal_adapter.get_size()

    @property
    def batch_size(self):
        return self._internal_adapter.batch_size

    def has_partial_batch(self):
        return self._internal_adapter.has_partial_batch()

    @property
    def partial_batch_size(self):
        return self._internal_adapter.partial_batch_size

    def should_recreate_iterator(self):
        return True
