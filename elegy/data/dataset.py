import numpy as np
import jax.numpy as jnp
import multiprocessing.pool
import typing as tp
from .data_adapter import DataAdapter
from .utils import is_none_or_empty

#TODO: typing
#TODO: docs
#TODO: mkdocs



class Dataset:
    def __getitem__(self, i):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError



class DataLoader:
    #TODO: __getitem__  incl slicing e.g. [:5]
    #TODO: custom batch_fn parameter
    #TODO: n_workers='auto'
    #TODO: worker_type
    #TODO: prefetch
    #TODO: timeout

    def __init__(self, dataset, batch_size, n_workers=0, shuffle=False, worker_type='thread'):
        assert batch_size>0 and type(batch_size)==int, 'batch_size must be a positive integer'
        assert worker_type in ['thread', 'process', 'spawn', 'fork', 'forkserver']

        self.dataset    = dataset
        self.batch_size = batch_size
        self.n_workers  = n_workers
        self.shuffle    = shuffle
        self.worker_type = worker_type
    
    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)
        
        batched_indices = [indices[i:][:self.batch_size] for i in range(0,len(indices),self.batch_size)]
        
        if self.n_workers==0:
            return mainthread_data_iterator(self.dataset, batched_indices)
        else:
            return multiprocess_data_iterator(self.dataset, batched_indices, self.n_workers, worker_type=self.worker_type)
    
    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))
    


def default_batch_fn(list_of_samples):
    assert len(list_of_samples)>0
    first_sample = list_of_samples[0]
    if hasattr(first_sample, '__array__'):
        return jnp.stack(list_of_samples)
    elif isinstance(first_sample, (tp.Tuple, tp.List)):
        sample_len     = len(first_sample)
        batched_lists  = [[sample[i] for sample in list_of_samples] for i in range(sample_len)]
        batched_stacks = [jnp.stack(batch) for batch in batched_lists]
        return tuple(batched_stacks)
    else:
        return tuple(list_of_samples)


def mainthread_data_iterator(ds, batched_indices):
    for batch_of_indices in batched_indices:
        samples = list(map(ds.__getitem__, batch_of_indices))
        yield default_batch_fn(samples)


def multiprocess_data_iterator(ds, batched_indices, n_workers, prefetch=1, timeout=10, worker_type='thread'):
    if worker_type=='thread':
        pool_class = multiprocessing.pool.ThreadPool
    else:
        worker_type = None if worker_type=='process' else worker_type #None means default
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
    

    def __init__(self, x: DataLoader, y=None, sample_weights=None,  **kwargs,):
        #shuffling is performed in the DataLoader
        kwargs.pop("shuffle", None)

        if not is_none_or_empty(y):
            raise ValueError(
                "`y` argument is not supported when using DataLoader as input. The underlying Dataset should return the y values."
            )
        if not is_none_or_empty(sample_weights):
            raise ValueError("`sample_weight` argument is not supported when using DataLoader as input. The underlying Dataset should return the sample weights."
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
    
