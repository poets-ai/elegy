import numpy as np
import jax.numpy as jnp
import multiprocessing.pool
import typing as tp
from .data_adapter import DataAdapter
from .utils import is_none_or_empty


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


class DataLoader:
    """Loads samples from a dataset and combines them into batches. Can be directly passed to `Model.fit()`"""  # +_example_usage_docstring

    # TODO: __getitem__  incl slicing e.g. [:5]
    # TODO: custom batch_fn parameter
    # TODO: n_workers='auto'
    # TODO: prefetch parameter
    # TODO: timeout parameter

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        n_workers: int = 0,
        shuffle: bool = False,
        worker_type: str = "thread",
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
        """
        assert (
            batch_size > 0 and type(batch_size) == int
        ), "batch_size must be a positive integer"
        assert worker_type in ["thread", "process", "spawn", "fork", "forkserver"]

        self.dataset = dataset
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.shuffle = shuffle
        self.worker_type = worker_type

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
            return multiprocess_data_iterator(
                self.dataset,
                batched_indices,
                self.n_workers,
                worker_type=self.worker_type,
            )

    def __len__(self) -> int:
        """Returns the number of batches per epoch"""
        return int(np.ceil(len(self.dataset) / self.batch_size))


Dataset.__doc__ += _example_usage_docstring
DataLoader.__doc__ += _example_usage_docstring


def default_batch_fn(
    list_of_samples: tp.List[tp.Any],
) -> tp.Union[jnp.ndarray, tp.Tuple[jnp.ndarray]]:
    """Batches individual data samples."""
    assert len(list_of_samples) > 0
    first_sample = list_of_samples[0]
    if hasattr(first_sample, "__array__"):
        return jnp.asarray(list_of_samples)
    elif isinstance(first_sample, (tp.Tuple, tp.List)):
        sample_len = len(first_sample)
        batched_lists = [
            [sample[i] for sample in list_of_samples] for i in range(sample_len)
        ]
        batched_stacks = [jnp.asarray(batch) for batch in batched_lists]
        return tuple(batched_stacks)
    else:
        return tuple(list_of_samples)


def mainthread_data_iterator(
    ds: Dataset, batched_indices: tp.List[tp.List[int]]
) -> tp.Iterable[tp.Any]:
    """Generator that loads datasamples from the data set in the main thread"""
    for batch_of_indices in batched_indices:
        samples = list(map(ds.__getitem__, batch_of_indices))
        yield default_batch_fn(samples)


def multiprocess_data_iterator(
    ds: Dataset,
    batched_indices: tp.List[tp.List[int]],
    n_workers: int,
    prefetch: int = 1,
    timeout: int = 10,
    worker_type: str = "thread",
) -> tp.Iterable[tp.Any]:
    """Generator that starts a pool of workers to load data samples from the dataset in parallel."""
    if worker_type == "thread":
        pool_class = multiprocessing.pool.ThreadPool
    else:
        worker_type = (
            None if worker_type == "process" else worker_type
        )  # None means default
        pool_class = multiprocessing.get_context(worker_type).Pool
    with pool_class(processes=n_workers) as pool:
        async_results = []
        for batch_of_indices in batched_indices[:prefetch]:
            async_results.append(pool.map_async(ds.__getitem__, batch_of_indices))

        for batch_of_indices in batched_indices[prefetch:]:
            async_results.append(pool.map_async(ds.__getitem__, batch_of_indices))
            samples = async_results.pop(0).get(timeout)
            yield default_batch_fn(samples)

        for async_result in async_results:
            samples = async_result.get(timeout)
            yield default_batch_fn(samples)


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
