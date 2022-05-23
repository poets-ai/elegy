import typing as tp
from typing import Callable

import jax.numpy as jnp

from elegy.callbacks.callback import Callback

"""
Methods:
--------
on_epoch_begin
on_epoch_end
on_train_batch_begin
on_train_batch_end
on_test_batch_begin
on_test_batch_end
on_predict_batch_begin
on_predict_batch_end
on_train_begin
on_train_end
on_test_begin
on_test_end
on_predict_begin
on_predict_end
"""


class ModuleCallback(Callback):
    def on_epoch_begin(
        self, epoch: int, logs: tp.Optional[tp.Dict[str, jnp.ndarray]] = None
    ):
        assert self.model is not None
        self.model.module = self.model.module.on_epoch_begin(epoch, logs)

    def on_epoch_end(
        self, epoch: int, logs: tp.Optional[tp.Dict[str, jnp.ndarray]] = None
    ):
        assert self.model is not None
        self.model.module = self.model.module.on_epoch_end(epoch, logs)

    def on_train_batch_begin(
        self, batch: int, logs: tp.Optional[tp.Dict[str, jnp.ndarray]] = None
    ):
        assert self.model is not None
        self.model.module = self.model.module.on_train_batch_begin(batch, logs)

    def on_train_batch_end(
        self, batch: int, logs: tp.Optional[tp.Dict[str, jnp.ndarray]] = None
    ):
        assert self.model is not None
        self.model.module = self.model.module.on_train_batch_end(batch, logs)

    def on_test_batch_begin(
        self, batch: int, logs: tp.Optional[tp.Dict[str, jnp.ndarray]] = None
    ):
        assert self.model is not None
        self.model.module = self.model.module.on_test_batch_begin(batch, logs)

    def on_test_batch_end(
        self, batch: int, logs: tp.Optional[tp.Dict[str, jnp.ndarray]] = None
    ):
        assert self.model is not None
        self.model.module = self.model.module.on_test_batch_end(batch, logs)

    def on_predict_batch_begin(
        self, batch: int, logs: tp.Optional[tp.Dict[str, jnp.ndarray]] = None
    ):
        assert self.model is not None
        self.model.module = self.model.module.on_predict_batch_begin(batch, logs)

    def on_predict_batch_end(
        self, batch: int, logs: tp.Optional[tp.Dict[str, jnp.ndarray]] = None
    ):
        assert self.model is not None
        self.model.module = self.model.module.on_predict_batch_end(batch, logs)

    def on_train_begin(self, logs: tp.Optional[tp.Dict[str, jnp.ndarray]] = None):
        assert self.model is not None
        self.model.module = self.model.module.on_train_begin(logs)

    def on_train_end(self, logs: tp.Optional[tp.Dict[str, jnp.ndarray]] = None):
        assert self.model is not None
        self.model.module = self.model.module.on_train_end(logs)

    def on_test_begin(self, logs: tp.Optional[tp.Dict[str, jnp.ndarray]] = None):
        assert self.model is not None
        self.model.module = self.model.module.on_test_begin(logs)

    def on_test_end(self, logs: tp.Optional[tp.Dict[str, jnp.ndarray]] = None):
        assert self.model is not None
        self.model.module = self.model.module.on_test_end(logs)

    def on_predict_begin(self, logs: tp.Optional[tp.Dict[str, jnp.ndarray]] = None):
        assert self.model is not None
        self.model.module = self.model.module.on_predict_begin(logs)

    def on_predict_end(self, logs: tp.Optional[tp.Dict[str, jnp.ndarray]] = None):
        assert self.model is not None
        self.model.module = self.model.module.on_predict_end(logs)
