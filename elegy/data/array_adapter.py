import math
import jax.numpy as jnp
import numpy as np
import typing as tp

from .data_adapter import DataAdapter
from .utils import pack_x_y_sample_weight


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
