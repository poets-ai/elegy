import typing as tp
from abc import abstractmethod

import jax
import jax.numpy as jnp
import jax_metrics as jm
import optax

import elegy as eg
from elegy import types
from elegy.data import unpack_x_y_sample_weight
from elegy.modules.managed.managed_module import ManagedModule
from elegy.optimizer import Optimizer

M = tp.TypeVar("M", bound="HighLevelModule")


class HighLevelModule(ManagedModule):
    losses_and_metrics: tp.Optional[jm.LossesAndMetrics]

    def __init__(
        self,
        *,
        losses_and_metrics: tp.Optional[jm.LossesAndMetrics] = None,
        optimizer: tp.Optional[
            tp.Union[optax.GradientTransformation, Optimizer]
        ] = None,
        initialized: bool = False,
        strategy: tp.Optional[tp.Union[str, "eg.Strategy"]] = None,
    ) -> None:
        super().__init__(
            optimizer=optimizer, initialized=initialized, strategy=strategy
        )
        self.losses_and_metrics = losses_and_metrics

    # ---------------------------------------------------------------------------
    # HIGH LEVEL API
    # ---------------------------------------------------------------------------
    @abstractmethod
    def init(self: M, key: jnp.ndarray, inputs: tp.Any) -> M:
        ...

    @abstractmethod
    def apply(
        self: M,
        key: jnp.ndarray,
        inputs: tp.Any,
        training: bool,
    ) -> tp.Tuple[types.Outputs, M]:
        ...

    def get_aux_losses(self) -> types.Logs:
        return {}

    def get_aux_metrics(self) -> types.Logs:
        return {}

    # ---------------------------------------------------------------------------
    # Managed API Implementation
    # ---------------------------------------------------------------------------

    def managed_init_step(self: M, key: jnp.ndarray, batch: tp.Any) -> M:
        if self.losses_and_metrics is None:
            raise RuntimeError(
                "Trying to run `managed_init_step` but no `losses_and_metrics` was given"
            )

        inputs, labels, sample_weight = unpack_x_y_sample_weight(batch)

        self = self.init(key, inputs)

        assert self.losses_and_metrics is not None
        self = self.replace(
            losses_and_metrics=self.losses_and_metrics.init(
                aux_losses=self.get_aux_losses(),
                aux_metrics=self.get_aux_metrics(),
            )
        )

        return self

    def managed_predict_step(
        self: M,
        key: jnp.ndarray,
        batch: tp.Any,
        batch_idx: jnp.ndarray,
    ) -> tp.Tuple[types.Outputs, M]:
        inputs, labels, sample_weight = unpack_x_y_sample_weight(batch)
        return self.apply(key, inputs, training=False)

    def _high_level_step(
        self: M,
        key: jnp.ndarray,
        batch: tp.Any,
        batch_idx: jnp.ndarray,
        epoch_idx: tp.Optional[jnp.ndarray],
        training: bool,
    ) -> tp.Tuple[types.Loss, M]:
        if self.losses_and_metrics is None:
            raise RuntimeError(
                "Trying to run `managed_train_step` no `losses_and_metrics`"
            )

        seq_key, pred_key = jax.random.split(key, 2)
        inputs, labels, sample_weight = unpack_x_y_sample_weight(batch)

        preds, self = self.apply(pred_key, inputs, training=training)

        if not isinstance(labels, tp.Mapping):
            label_args = dict(target=labels)
        else:
            label_args = labels

        if not isinstance(inputs, tp.Mapping):
            input_args = dict(inputs=inputs)
        else:
            input_args = inputs

        assert self.losses_and_metrics is not None

        batch_updates = self.losses_and_metrics.batch_updates(
            preds=preds,
            module=self,
            parameters=self._get_params(),
            batch_stats=self._get_batch_stats(),
            key_seq=types.KeySeq(seq_key),
            aux_losses=self.get_aux_losses(),
            aux_metrics=self.get_aux_metrics(),
            batch_idx=batch_idx,
            epoch_idx=epoch_idx,
            sample_weight=sample_weight,
            **label_args,
            **input_args,
        )

        loss = batch_updates.total_loss()

        self = self.log("losses_and_metrics", batch_updates)

        return loss, self

    def managed_train_step(
        self: M,
        key: jnp.ndarray,
        batch: tp.Any,
        batch_idx: jnp.ndarray,
        epoch_idx: jnp.ndarray,
    ) -> tp.Tuple[types.Loss, M]:
        return self._high_level_step(key, batch, batch_idx, epoch_idx, training=True)

    def managed_test_step(
        self: M, key: jnp.ndarray, batch: tp.Any, batch_idx: jnp.ndarray
    ) -> M:
        loss, self = self._high_level_step(
            key, batch, batch_idx, epoch_idx=None, training=False
        )

        return self
