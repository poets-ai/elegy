# Implementation based on tf.keras.engine.data_adapter.py
# https://github.com/tensorflow/tensorflow/blob/2b96f3662bd776e277f86997659e61046b56c315/tensorflow/python/keras/engine/data_adapter.py


import math
import typing as tp
from operator import itemgetter

import jax.numpy as jnp
import numpy as np
from elegy import types

from .data_adapter import DataAdapter
from .utils import flatten, map_structure, pack_x_y_sample_weight

DEFAULT_BATCH_SIZE = 32


class ArrayDataAdapter(DataAdapter):
    """Adapter that handles NumPy and Jax numpy arrays."""

    @staticmethod
    def can_handle(x, y=None):
        flat_inputs = list(flatten(x))
        if y is not None:
            flat_inputs += list(flatten(y))

        supported_types = (jnp.ndarray, np.ndarray)
        # if pd:
        #     supported_types = (ops.Tensor, np.ndarray, pd.Series, pd.DataFrame)

        def _is_array(v):
            if isinstance(v, supported_types):
                return True
            return False

        return all(_is_array(v) for v in flat_inputs)

    def __init__(
        self,
        x: types.ArrayHolder,
        y: tp.Union[types.ArrayHolder, None] = None,
        sample_weights: tp.Union[jnp.ndarray, np.ndarray, None] = None,
        batch_size: tp.Optional[int] = None,
        epochs: int = 1,
        steps: tp.Optional[int] = None,
        shuffle: bool = False,
        drop_remainder: bool = False,
        **kwargs,
    ):
        super(ArrayDataAdapter, self).__init__(x, y, **kwargs)

        inputs = pack_x_y_sample_weight(x, y, sample_weights)

        num_samples = set(int(i.shape[0]) for i in flatten(inputs))

        if len(num_samples) > 1:
            msg = "Data cardinality is ambiguous:\n"
            for label, data in zip(["x", "y", "sample_weight"], inputs):
                msg += "  {} sizes: {}\n".format(
                    label, ", ".join(str(i.shape[0]) for i in data)
                )
            msg += "Please provide data which shares the same first dimension."
            raise ValueError(msg)

        num_samples = (
            num_samples.pop()
            if num_samples
            else batch_size
            if batch_size is not None
            else DEFAULT_BATCH_SIZE
        )

        # If batch_size is not passed but steps is, calculate from the input data.
        if batch_size is None:
            # if batch_size is None and steps is None:
            #     raise ValueError("Please provide either batch_size or steps")
            batch_size = (
                int(math.ceil(num_samples / steps)) if steps else DEFAULT_BATCH_SIZE
            )

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
                    #     print("Dropping!")
                    #     continue
                    inputs_slices = map_structure(itemgetter(indices), inputs)

                    yield inputs_slices

        self._dataset = dataset_generator

    def get_dataset(self):
        return self._dataset

    def get_size(self):
        return self._size

    @property
    def batch_size(self):
        return self._batch_size

    def has_partial_batch(self):
        return self._partial_batch_size > 0

    @property
    def partial_batch_size(self):
        return self._partial_batch_size or None

    def should_recreate_iterator(self):
        # An infinite dataset is always created here.
        return False
