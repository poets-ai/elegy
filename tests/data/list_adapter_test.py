import math
from unittest import TestCase

import numpy as np
import jax.numpy as jnp

import pytest

from elegy.data.list_adapter import ListsOfScalarsDataAdapter


class ListsOfScalarsDataAdapterTest(TestCase):
    def test_basic(self):
        x = np.random.uniform(size=(100, 32, 32, 3))
        y = np.random.uniform(size=(100, 1))
        batch_size = 10
        epochs = 1
        data_adapter = ListsOfScalarsDataAdapter(
            x.tolist(),
            y=y.tolist(),
            sample_weights=None,
            batch_size=batch_size,
            epochs=epochs,
            steps=None,
            shuffle=False,
        )
        num_steps = math.ceil(x.shape[0] / batch_size) * epochs
        iterator_fn = data_adapter.get_dataset()
        for i, batch in zip(range(num_steps), iterator_fn()):
            batch_x, batch_y = batch
            assert batch_x.shape == (batch_size, *x.shape[1:])
            assert batch_y.shape == (batch_size, *y.shape[1:])
            np.testing.assert_array_equal(
                batch_x, x[i * batch_size : (i + 1) * batch_size]
            )

        data_adapter.get_size() == x.shape[0]
        data_adapter.partial_batch_size == 0

    def test_jax(self):
        x = jnp.array(np.random.uniform(size=(100, 32, 32, 3)))
        y = jnp.array(np.random.uniform(size=(100, 1)))
        batch_size = 10
        epochs = 1
        data_adapter = ListsOfScalarsDataAdapter(
            x.tolist(),
            y=y.tolist(),
            sample_weights=None,
            batch_size=batch_size,
            epochs=epochs,
            steps=None,
            shuffle=False,
        )
        num_steps = math.ceil(x.shape[0] / batch_size) * epochs
        iterator_fn = data_adapter.get_dataset()
        for i, batch in zip(range(num_steps), iterator_fn()):
            batch_x, batch_y = batch
            assert batch_x.shape == (batch_size, *x.shape[1:])
            assert batch_y.shape == (batch_size, *y.shape[1:])
            np.testing.assert_array_equal(
                batch_x, x[i * batch_size : (i + 1) * batch_size]
            )

        data_adapter.get_size() == x.shape[0]
        data_adapter.partial_batch_size == 0

    def test_shuffle(self):
        x = np.random.uniform(size=(100, 32, 32, 3))
        y = np.random.uniform(size=(100, 1))
        batch_size = 10
        epochs = 1
        data_adapter = ListsOfScalarsDataAdapter(
            x.tolist(),
            y=y.tolist(),
            sample_weights=None,
            batch_size=batch_size,
            epochs=epochs,
            steps=None,
            shuffle=True,
        )
        num_steps = math.ceil(x.shape[0] / batch_size) * epochs
        iterator_fn = data_adapter.get_dataset()
        for i, batch in zip(range(num_steps), iterator_fn()):
            batch_x, batch_y = batch
            assert batch_x.shape == (batch_size, *x.shape[1:])
            assert batch_y.shape == (batch_size, *y.shape[1:])
            assert not np.array_equal(batch_x, x[i * batch_size : (i + 1) * batch_size])

        data_adapter.get_size() == x.shape[0]
        data_adapter.partial_batch_size == 0

    def test_partial_batch(self):
        x = np.random.uniform(size=(100, 32, 32, 3))
        y = np.random.uniform(size=(100, 1))
        batch_size = 32
        epochs = 1
        data_adapter = ListsOfScalarsDataAdapter(
            x.tolist(),
            y=y.tolist(),
            sample_weights=None,
            batch_size=batch_size,
            epochs=epochs,
            steps=None,
            shuffle=True,
        )
        num_steps = math.ceil(x.shape[0] / batch_size) * epochs

        iterator_fn = data_adapter.get_dataset()
        for i, batch in zip(range(num_steps), iterator_fn()):
            batch_x, batch_y = batch
            if i < num_steps - 1:
                assert batch_x.shape == (batch_size, *x.shape[1:])
                assert batch_y.shape == (batch_size, *y.shape[1:])
            else:
                assert batch_x.shape == (x.shape[0] % batch_size, *x.shape[1:])
                assert batch_y.shape == (x.shape[0] % batch_size, *y.shape[1:])

        data_adapter.get_size() == x.shape[0]
        data_adapter.partial_batch_size == x.shape[0] % batch_size
