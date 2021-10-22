import math
from unittest import TestCase

import numpy as np
import torch
from elegy.data.torch_dataloader_adapter import TorchDataLoaderAdapter
from torch.utils.data import DataLoader, TensorDataset


class ArrayDataAdapterTest(TestCase):
    def test_basic(self):
        batch_size = 10
        epochs = 1
        x = np.array(np.random.uniform(size=(100, 32, 32, 3)))
        y = np.array(np.random.uniform(size=(100, 1)))

        dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        dataloader = DataLoader(dataset, batch_size=batch_size)

        data_adapter = TorchDataLoaderAdapter(dataloader)

        dataset_length = x.shape[0]
        num_steps = math.ceil(dataset_length / batch_size) * epochs
        iterator_fn = data_adapter.get_dataset()
        for i, batch in zip(range(num_steps), iterator_fn()):
            batch_x, batch_y = batch
            assert batch_x.shape == (batch_size, *x.shape[1:])
            assert batch_y.shape == (batch_size, *y.shape[1:])
            np.testing.assert_array_equal(
                batch_x,
                x[
                    (i * batch_size)
                    % dataset_length : (i * batch_size)
                    % dataset_length
                    + batch_size
                ],
            )

        assert data_adapter.get_size() * batch_size == x.shape[0]
        assert data_adapter.batch_size == batch_size
        assert i == num_steps - 1
