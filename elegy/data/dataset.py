import numpy as np
import jax, jax.numpy as jnp
import multiprocessing.pool
import typing as tp
from .data_adapter import DataAdapter
from .utils import is_none_or_empty
import os


__all__ = ["Dataset", "DataLoader"]


_example_usage_docstring = """

Example Usage:
```
class MyDataset(elegy.data.Dataset):
def __len__(self):
    return 128

def __getitem__(self, i):
    #dummy data
    return np.random.random([224, 224, 3]),  np.random.randint(10)

ds     = MyDataset()
loader = elegy.data.DataLoader(ds, batch_size=8, n_workers=8, worker_type='thread', shuffle=True)
model.fit(loader, epochs=10)
```
"""


class Dataset:
    """Abstract base class for datasets. Subclasses should implement the `__getitem__` and `__len__` methods."""  # +_example_usage_docstring

    __all__ = ["__getitem__", "__len__"]

    def __getitem__(self, i: int) -> tp.Any:
        """Abstract method. In a subclass this should return the `i`-th data sample"""
        raise NotImplementedError

    def __len__(self) -> int:
        """Abstract method. In a subclass this should return the number of data samples in the dataset."""
        raise NotImplementedError

    def batch_fn(
        self, list_of_samples: tp.List[tp.Any]
    ) -> tp.Union[jnp.ndarray, tp.Tuple[jnp.ndarray]]:
        """Used by DataLoader to group a list of individual samples into a batch.
        By default tries to stack elements in the samples according to their positiion.
        Can be overridden for more complex use cases.
        """
        return default_batch_fn(list_of_samples)


class DataLoader:
    """Loads samples from a dataset and combines them into batches. Can be directly passed to `Model.fit()`"""  # +_example_usage_docstring

    # TODO: __getitem__  incl slicing e.g. [:5]
    # TODO: n_workers='auto'
    # TODO: timeout parameter

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        n_workers: tp.Optional[int] = 0,
        shuffle: tp.Optional[bool] = False,
        worker_type: tp.Optional[str] = "thread",
        prefetch: tp.Optional[int] = 1,
    ):
        """
        Arguments:
            dataset: The dataset from which to load samples.
                     A subclass of elegy.data.Dataset or an iterable which implements `__getitem__` and `__len__`.
            batch_size: A positive integer specifying how many samples a batch should have.
            n_workers: The number of parallel worker threads or processes which load data from the dataset.
                       A value of 0 (default) means to load data from the main thread.
            shuffle: Whether to load the samples in random order or not. Reshuffles on every epoch if True. Default: False
            worker_type: One of 'thread' (default), 'process', 'spawn', 'fork' or 'forkserver'. Only used if `n_workers`>0.
                         Threads are light-weight but underly the limitations of Python's global interpreter lock.
                         'process' uses the default process type as defined in the `multiprocessing` module.
                         'spawn', 'fork' and 'forkserver' can be used to select a specific process type.
                         For more information consult the Python `multiprocessing` documentation.
            prefetch: Number of batches to prefetch for pipelined execution (Default: 2)
        """
        assert (
            batch_size > 0 and type(batch_size) == int
        ), "batch_size must be a positive integer"
        assert worker_type in ["thread", "process", "spawn", "fork", "forkserver"]
        assert (
            prefetch >= 0 and type(prefetch) == int
        ), "prefetch must be a non-negative integer"

        self.dataset = dataset
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.shuffle = shuffle
        self.worker_type = worker_type
        self.prefetch = prefetch

    def __len__(self) -> int:
        """Returns the number of batches per epoch"""
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def __iter__(self) -> tp.Generator[tp.Any, None, None]:
        """Returns a generator which generates batches of loaded data samples"""
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)

        batched_indices = [
            indices[i:][: self.batch_size]
            for i in range(0, len(indices), self.batch_size)
        ]

        if self.n_workers == 0:
            return mainthread_data_iterator(self.dataset, batched_indices)
        else:
            return MultiProcessIterator(
                self.dataset,
                batched_indices,
                self.n_workers,
                prefetch=self.prefetch,
                worker_type=self.worker_type,
            )


Dataset.__doc__ += _example_usage_docstring
DataLoader.__doc__ += _example_usage_docstring


def default_batch_fn(
    list_of_samples: tp.List[tp.Any],
) -> tp.Union[jnp.ndarray, tp.Tuple[jnp.ndarray]]:
    """Batches individual data samples."""
    assert len(list_of_samples) > 0
    return jax.tree_multimap(lambda *x: jnp.asarray(x), *list_of_samples)


def get_batch_fn(ds: tp.Any) -> tp.Callable:
    """Returns either the batch_fn of the argument if it has one, otherwise `default_batch_fn`
    to allow arrays or datasets that don't inherit from elegy.data.Dataset"""
    return getattr(ds, "batch_fn", default_batch_fn)


def mainthread_data_iterator(
    ds: Dataset, batched_indices: tp.List[tp.List[int]]
) -> tp.Iterable[tp.Any]:
    """Generator that loads datasamples from the data set in the main thread"""
    for batch_of_indices in batched_indices:
        samples = list(map(ds.__getitem__, batch_of_indices))
        yield get_batch_fn(ds)(samples)


class WorkerContext:
    """A namespace to store the Dataset object for each process
    instead of passing it in each iteration which causes re-pickling"""

    _per_process_data = dict()

    @classmethod
    def init(cls, ds, worker_type):
        cls._per_process_data[os.getpid()] = ds
        if worker_type != "thread":
            # disable keyboard interrupts in worker process
            import signal

            signal.signal(signal.SIGINT, signal.SIG_IGN)

    @classmethod
    def get_sample(cls, i):
        ds = cls._per_process_data[os.getpid()]
        return ds[i]


class MultiProcessIterator:
    def __init__(
        self,
        ds: Dataset,
        batched_indices: tp.List[tp.List[int]],
        n_workers: int,
        prefetch: int = 2,
        timeout: int = 10,
        worker_type: str = "thread",
    ):
        self.ds = ds
        self.batched_indices = batched_indices
        self.timeout = timeout

        if worker_type == "thread":
            pool_class = multiprocessing.pool.ThreadPool
        else:
            worker_type = (
                None if worker_type == "process" else worker_type
            )  # None means default
            pool_class = multiprocessing.get_context(worker_type).Pool
        self.worker_pool = pool_class(
            n_workers, initializer=WorkerContext.init, initargs=(ds, worker_type)
        )
        # extra thread to transfer data to the device
        self.data_transfer_worker = multiprocessing.pool.ThreadPool(processes=1)

        self.async_results_queue = []
        for batch_of_indices in batched_indices[:prefetch]:
            self.dispatch_tasks(batch_of_indices)
        self.batched_indices = batched_indices[prefetch:]

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.batched_indices):
            batch_of_indices = self.batched_indices.pop(0)
            self.dispatch_tasks(batch_of_indices)

        if len(self.async_results_queue) == 0:
            raise StopIteration
        async_x = self.async_results_queue.pop(0)

        try:
            x = async_x.get(timeout=self.timeout)
        except KeyboardInterrupt:
            self.shutdown()
            raise
        batch = x
        return batch

    def dispatch_tasks(self, batch_of_indices):
        async_x = self.worker_pool.map_async(WorkerContext.get_sample, batch_of_indices)
        async_x = self.data_transfer_worker.apply_async(
            self.data_transfer_fn, (async_x,)
        )
        self.async_results_queue.append(async_x)

    def shutdown(self):
        self.worker_pool.close()
        for a_result in self.async_results_queue:
            # wait for remaining tasks to finish
            # process workers will hang otherwise
            a_result.wait(timeout=self.timeout)
        self.worker_pool.terminate()
        self.worker_pool.join()

    def __del__(self):
        self.shutdown()

    def data_transfer_fn(self, async_map_result, timeout=10):
        samples = async_map_result.get(timeout)
        batch = get_batch_fn(self.ds)(samples)
        # make sure the batch is transferred to the device
        batch = jax.tree_map(jnp.asarray, batch)
        return batch


class DataLoaderAdapter(DataAdapter):
    @staticmethod
    def can_handle(x, y=None):
        return isinstance(x, DataLoader)

    def __init__(
        self,
        x: DataLoader,
        y=None,
        sample_weights=None,
        **kwargs,
    ):
        # shuffling is performed in the DataLoader
        kwargs.pop("shuffle", None)

        if not is_none_or_empty(y):
            raise ValueError(
                "`y` argument is not supported when using DataLoader as input. The underlying Dataset should return the y values."
            )
        if not is_none_or_empty(sample_weights):
            raise ValueError(
                "`sample_weight` argument is not supported when using DataLoader as input. The underlying Dataset should return the sample weights."
            )

        super().__init__(x, y, **kwargs)
        self._dataloader = x

    def should_recreate_iterator(self):
        return True

    def get_dataset(self):
        dataloader = self._dataloader

        def dataloader_wrapper():
            yield from dataloader

        return dataloader_wrapper

    def get_size(self):
        return len(self._dataloader)

    @property
    def batch_size(self):
        return self.representative_batch_size

    @property
    def representative_batch_size(self):
        return self._dataloader.batch_size

    def has_partial_batch(self):
        return False

    @property
    def partial_batch_size(self):
        return
