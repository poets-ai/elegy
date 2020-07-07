import math
from unittest import TestCase

import numpy as np
import jax.numpy as jnp

import pytest

# from elegy.data_adapter import ArrayDataAdapter


class ArrayDataAdapterTest(TestCase):
    def test_basic(self):
        x = np.random.uniform(size=(100, 32, 32, 3))
        y = np.random.uniform(size=(100, 1))
        batch_size = 10
        epochs = 1
        data_adapter = ArrayDataAdapter(
            x,
            y=y,
            sample_weights=None,
            batch_size=batch_size,
            epochs=epochs,
            steps=None,
            shuffle=False,
        )
        num_steps = math.ceil(x.shape[0] / batch_size) * epochs

        for i, batch in enumerate(data_adapter.get_dataset()):
            batch_x, batch_y = batch
            assert batch_x.shape == (batch_size, *x.shape[1:])
            assert batch_y.shape == (batch_size, *y.shape[1:])
            np.testing.assert_array_equal(
                batch_x, x[i * batch_size : (i + 1) * batch_size]
            )

        assert i + 1 == num_steps

    def test_jax(self):
        x = jnp.array(np.random.uniform(size=(100, 32, 32, 3)))
        y = jnp.array(np.random.uniform(size=(100, 1)))
        batch_size = 10
        epochs = 1
        data_adapter = ArrayDataAdapter(
            x,
            y=y,
            sample_weights=None,
            batch_size=batch_size,
            epochs=epochs,
            steps=None,
            shuffle=False,
        )
        num_steps = math.ceil(x.shape[0] / batch_size) * epochs

        for i, batch in enumerate(data_adapter.get_dataset()):
            batch_x, batch_y = batch
            assert batch_x.shape == (batch_size, *x.shape[1:])
            assert batch_y.shape == (batch_size, *y.shape[1:])
            np.testing.assert_array_equal(
                batch_x, x[i * batch_size : (i + 1) * batch_size]
            )

        assert i + 1 == num_steps

    def test_shuffle(self):
        x = np.random.uniform(size=(100, 32, 32, 3))
        y = np.random.uniform(size=(100, 1))
        batch_size = 10
        epochs = 1
        data_adapter = ArrayDataAdapter(
            x,
            y=y,
            sample_weights=None,
            batch_size=batch_size,
            epochs=epochs,
            steps=None,
            shuffle=True,
        )
        num_steps = math.ceil(x.shape[0] / batch_size) * epochs

        for i, batch in enumerate(data_adapter.get_dataset()):
            batch_x, batch_y = batch
            assert batch_x.shape == (batch_size, *x.shape[1:])
            assert batch_y.shape == (batch_size, *y.shape[1:])
            assert not np.array_equal(batch_x, x[i * batch_size : (i + 1) * batch_size])

        assert i + 1 == num_steps

    def test_batch_size(self):
        x = np.random.uniform(size=(100, 32, 32, 3))
        y = np.random.uniform(size=(100, 1))
        batch_size = 32
        epochs = 1
        data_adapter = ArrayDataAdapter(
            x,
            y=y,
            sample_weights=None,
            batch_size=batch_size,
            epochs=epochs,
            steps=None,
            shuffle=True,
        )
        num_steps = math.ceil(x.shape[0] / batch_size) * epochs

        for i, batch in enumerate(data_adapter.get_dataset()):
            batch_x, batch_y = batch
            assert batch_x.shape == (batch_size, *x.shape[1:])
            assert batch_y.shape == (batch_size, *y.shape[1:])

        assert i + 1 == num_steps

    def test_epochs(self):
        x = np.random.uniform(size=(100, 32, 32, 3))
        y = np.random.uniform(size=(100, 1))
        batch_size = 32
        epochs = 3
        data_adapter = ArrayDataAdapter(
            x,
            y=y,
            sample_weights=None,
            batch_size=batch_size,
            epochs=epochs,
            steps=None,
            shuffle=True,
        )
        num_steps = math.ceil(x.shape[0] / batch_size) * epochs

        for i, batch in enumerate(data_adapter.get_dataset()):
            batch_x, batch_y = batch
            assert batch_x.shape == (batch_size, *x.shape[1:])
            assert batch_y.shape == (batch_size, *y.shape[1:])

        assert i + 1 == num_steps

