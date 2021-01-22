from jax._src.lax.lax import remaining
from torch.utils.data import DataLoader

from .data_adapter import DataAdapter
from .utils import is_none_or_empty, map_structure, list_to_tuple


class TorchDataLoaderAdapter(DataAdapter):
    """Adapter that handles torch Dataloaders."""

    @staticmethod
    def can_handle(x, y=None):
        return isinstance(x, DataLoader)

    def __init__(
        self,
        x: DataLoader,
        y=None,
        steps=None,
        sample_weights=None,
        training=True,
        **kwargs,
    ):

        if not is_none_or_empty(y):
            raise ValueError(
                "`y` argument is not supported when using " "torch Dataloader as input."
            )
        if not is_none_or_empty(sample_weights):
            raise ValueError(
                "`sample_weight` argument is not supported when using "
                "torch Dataloader as input."
            )

        super().__init__(x, y, **kwargs)

        self.training = training
        self.steps = steps
        self._batch_size = x.batch_size
        self._dataset = x

        self.current_step = 0

    def get_dataset(self):
        def parse_dataloader_gen():
            self.current_step = 0
            for batch in iter(self._dataset):
                self.current_step += 1
                batch = map_structure(lambda x: x.cpu().numpy(), list_to_tuple(batch))
                yield batch

        return parse_dataloader_gen

    def get_size(self):
        try:
            return len(self._dataset)
        except Exception:
            return None

    @property
    def batch_size(self):
        return self.representative_batch_size

    @property
    def representative_batch_size(self):
        return self._batch_size

    def has_partial_batch(self):
        return False

    @property
    def partial_batch_size(self):
        return

    def should_recreate_iterator(self):
        # if in eval mode should not recreate iterator
        # but if in train mode and steps not provided, should recreate at end of each epoch
        if not self.training or self.steps is None:
            return self.training

        steps_dataset = self.get_size()
        if steps_dataset is None:
            return False

        remaining_steps = steps_dataset - self.current_step
        # if remaining steps less than needed steps, should recreate dataloader
        # TODO: This will drop the last steps of data, how to avoid this?
        if remaining_steps < self.steps:
            return True
        else:
            return False
