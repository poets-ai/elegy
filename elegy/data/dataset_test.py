from unittest import TestCase
import numpy as np
import jax.numpy as jnp

import elegy, optax


class DataLoaderTestCase(TestCase):
    def test_basic(self):
        ds = DS0()
        assert len(ds) == 11

        loader = elegy.data.DataLoader(ds, batch_size=4, n_workers=0, shuffle=False)
        batches = list(loader)
        assert len(loader) == len(batches) == 3
        assert len(batches[0]) == 2

        batched_x = [x for x, y in batches]
        batched_y = [y for x, y in batches]

        assert (
            all(batched_y[0] == np.arange(4))
            and all(batched_y[1] == np.arange(4) + 4)
            and all(batched_y[2] == np.arange(3) + 8)
        )
        assert batched_x[0].shape == batched_x[1].shape == (4, 100, 200, 3)
        assert batched_x[2].shape == (3, 100, 200, 3)

    def test_multiprocessed(self):
        ds = DS0()
        loader = elegy.data.DataLoader(ds, batch_size=4, n_workers=4)
        batches = list(loader)
        assert len(loader) == len(batches) == 3
        assert len(batches[0]) == 2

        batched_x = [x for x, y in batches]
        batched_y = [y for x, y in batches]

        assert (
            all(batched_y[0] == np.arange(4))
            and all(batched_y[1] == np.arange(4) + 4)
            and all(batched_y[2] == np.arange(3) + 8)
        )
        assert batched_x[0].shape == batched_x[1].shape == (4, 100, 200, 3)
        assert batched_x[2].shape == (3, 100, 200, 3)

    def test_not_implemented_ds(self):
        class DS0(elegy.data.Dataset):
            ...

        class DS1(elegy.data.Dataset):
            def __len__(self):
                return 5

        for DS in [DS0, DS1]:
            ds = DS()
            loader = elegy.data.DataLoader(ds, 10)
            try:
                list(loader)
            except NotImplementedError:
                pass
            else:
                assert False

    def test_shuffled(self):
        ds = DS0()
        loader = elegy.data.DataLoader(ds, batch_size=4, n_workers=4, shuffle=True)
        batches0 = list(loader)
        batches1 = list(loader)

        batched_y0 = [y for x, y in batches0]
        batched_y1 = [y for x, y in batches1]

        assert not (
            all(batched_y0[0] == np.arange(4))
            and all(batched_y0[1] == np.arange(4) + 4)
            and all(batched_y0[2] == np.arange(3) + 8)
        )
        assert not (
            all(batched_y0[0] == batched_y1[0])
            and all(batched_y0[1] == batched_y1[1])
            and all(batched_y0[2] == batched_y1[2])
        )

    def test_model_fit(self):
        ds = DS0()
        loader_train = elegy.data.DataLoader(ds, batch_size=4, n_workers=4)
        loader_valid = elegy.data.DataLoader(ds, batch_size=4, n_workers=4)

        class Module(elegy.Module):
            def call(self, x):
                x = jnp.mean(x, axis=[1, 2])
                x = elegy.nn.Linear(20)(x)
                return x

        model = elegy.Model(
            Module(),
            loss=elegy.losses.SparseCategoricalCrossentropy(),
            optimizer=optax.sgd(0.1),
        )
        model.fit(loader_train, validation_data=loader_valid, epochs=3)

    def test_multi_item_ds(self):
        # one item
        class DS0:
            def __len__(self):
                return 11

            def __getitem__(self, i):
                return np.zeros([100, 200, 3])

        loader0 = elegy.data.DataLoader(DS0(), batch_size=4)
        batches0 = list(loader0)
        assert len(batches0) == 3
        assert (
            batches0[0].shape == (4, 100, 200, 3)
            and batches0[1].shape == (4, 100, 200, 3)
            and batches0[2].shape == (3, 100, 200, 3)
        )

        # three items
        class DS1:
            def __len__(self):
                return 11

            def __getitem__(self, i):
                return (
                    np.zeros([100, 200, 3]),
                    np.zeros([100, 200, 3]),
                    np.zeros([100, 200, 3]),
                )

        loader1 = elegy.data.DataLoader(DS1(), batch_size=4)
        batches1 = list(loader1)
        assert len(batches1) == 3
        assert isinstance(batches1[0], tuple)
        assert len(batches1[0]) == 3
        assert (
            batches1[0][0].shape == (4, 100, 200, 3)
            and batches1[1][1].shape == (4, 100, 200, 3)
            and batches1[2][2].shape == (3, 100, 200, 3)
        )

    def test_worker_type(self):
        ds = DS0()
        for worker_type in ["thread", "process", "spawn", "fork", "forkserver"]:
            loader = elegy.data.DataLoader(
                ds, batch_size=4, n_workers=4, worker_type=worker_type
            )
            batches = list(loader)

    def test_prefetch(self):
        ds = DS0()

        loader = elegy.data.DataLoader(
            ds, batch_size=2, n_workers=4, shuffle=False, prefetch=3
        )
        batches = list(loader)
        assert len(loader) == len(batches)

    def test_custom_batch_fn(self):
        ds = DS_custom_batch_fn()
        loader = elegy.data.DataLoader(ds, batch_size=3)
        batches = list(loader)
        assert len(loader) == len(batches)
        assert batches[0]["a"].shape == (3, 10)
        assert batches[0]["b"].shape == (3,)
        assert batches[0]["c"] == "banana"
        assert np.all(batches[0]["b"] == np.array([0, 1, 2]))

    def test_loader_from_array(self):
        pseudo_ds = np.arange(65)
        loader = elegy.data.DataLoader(pseudo_ds, batch_size=10)
        batches = list(loader)
        assert len(batches) == 7
        assert np.all(batches[1] == np.arange(10, 20))


class DS0(elegy.data.Dataset):
    def __len__(self):
        return 11

    def __getitem__(self, i):
        return np.zeros([100, 200, 3]), np.arange(20)[i]


class DS_custom_batch_fn(elegy.data.Dataset):
    def __len__(self):
        return 11

    def __getitem__(self, i):
        return dict(a=np.random.random(size=10), b=i)

    def batch_fn(self, list_of_samples):
        x = super().batch_fn(list_of_samples)
        x.update(c="banana")
        return x
