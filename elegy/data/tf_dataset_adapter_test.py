import math
from unittest import TestCase

import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from elegy.data.tf_dataset_adapter import TFDatasetAdapter


class ArrayDataAdapterTest(TestCase):
    def test_basic(self):
        batch_size = 10
        epochs = 1
        x = np.array(np.random.uniform(size=(100, 32, 32, 3)))
        y = np.array(np.random.uniform(size=(100, 1)))
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.batch(batch_size)

        data_adapter = TFDatasetAdapter(dataset, steps=None)

        num_steps = math.ceil(x.shape[0] / batch_size) * epochs
        iterator_fn = data_adapter.get_dataset()
        for i, batch in zip(range(num_steps), iterator_fn()):
            batch_x, batch_y = batch
            assert batch_x.shape == (batch_size, *x.shape[1:])
            assert batch_y.shape == (batch_size, *y.shape[1:])
            np.testing.assert_array_equal(
                batch_x, x[i * batch_size : (i + 1) * batch_size]
            )

        assert data_adapter.get_size() * batch_size == x.shape[0]
        assert data_adapter.batch_size == batch_size

    def test_only_x_repeat(self):
        batch_size = 10
        epochs = 2

        x = np.array(np.random.uniform(size=(100, 32, 32, 3)))
        dataset = tf.data.Dataset.from_tensor_slices(x)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()

        dataset_length = x.shape[0]
        num_steps = math.ceil(dataset_length / batch_size) * epochs

        data_adapter = TFDatasetAdapter(
            dataset, steps=math.ceil(dataset_length / batch_size)
        )

        iterator_fn = data_adapter.get_dataset()
        for i, batch in zip(range(num_steps), iterator_fn()):
            batch_x = batch
            assert batch_x.shape == (batch_size, *x.shape[1:])
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

    def test_error(self):
        batch_size = 10
        epochs = 2
        x = np.array(np.random.uniform(size=(100, 32, 32, 3)))
        dataset = tf.data.Dataset.from_tensor_slices(x)
        dataset = dataset.batch(batch_size)

        data_adapter = TFDatasetAdapter(dataset, steps=None)

        num_steps = math.ceil(x.shape[0] / batch_size) * epochs
        iterator_fn = data_adapter.get_dataset()
        iterator = iterator_fn()

        with self.assertRaises(StopIteration):
            for i in range(num_steps):
                batch = next(iterator)
                batch_x = batch
                assert batch_x.shape == (batch_size, *x.shape[1:])
                np.testing.assert_array_equal(
                    batch_x, x[i * batch_size : (i + 1) * batch_size]
                )
