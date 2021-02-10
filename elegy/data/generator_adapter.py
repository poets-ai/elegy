import itertools
import typing as tp

from .data_adapter import DataAdapter
from .utils import (
    assert_not_namedtuple,
    flatten,
    is_none_or_empty,
    pack_x_y_sample_weight,
    unpack_x_y_sample_weight,
)


class GeneratorDataAdapter(DataAdapter):
    """Adapter that handles python generators and iterators."""

    @staticmethod
    def can_handle(x, y=None):
        return (hasattr(x, "__next__") or hasattr(x, "next")) and hasattr(x, "__iter__")

    def __init__(
        self,
        x: tp.Union[tp.Iterable],
        y=None,
        sample_weights=None,
        **kwargs,
    ):
        # Generators should never shuffle as exhausting the generator in order to
        # shuffle the batches is inefficient.
        kwargs.pop("shuffle", None)

        if not is_none_or_empty(y):
            raise ValueError(
                "`y` argument is not supported when using " "python generator as input."
            )
        if not is_none_or_empty(sample_weights):
            raise ValueError(
                "`sample_weight` argument is not supported when using "
                "python generator as input."
            )

        super(GeneratorDataAdapter, self).__init__(x, y, **kwargs)

        # Since we have to know the dtype of the python generator when we build the
        # dataset, we have to look at a batch to infer the structure.
        peek, x = self._peek_and_restore(x)
        assert_not_namedtuple(peek)
        peek = self._standardize_batch(peek)

        self._first_batch_size = int(list(flatten(peek))[0].shape[0])

        def wrapped_generator():
            for data in x:
                yield self._standardize_batch(data)

        dataset = wrapped_generator

        self._dataset = dataset

    def _standardize_batch(self, data):
        """Standardizes a batch output by a generator."""
        # Removes `None`s.
        x, y, sample_weight = unpack_x_y_sample_weight(data)
        data = pack_x_y_sample_weight(x, y, sample_weight)

        return data

    @staticmethod
    def _peek_and_restore(x):
        peek = next(x)
        return peek, itertools.chain([peek], x)

    def get_dataset(self):
        return self._dataset

    def get_size(self):
        return None

    @property
    def batch_size(self):
        return self.representative_batch_size

    @property
    def representative_batch_size(self):
        return self._first_batch_size

    def has_partial_batch(self):
        return False

    @property
    def partial_batch_size(self):
        return

    def should_recreate_iterator(self):
        return False
