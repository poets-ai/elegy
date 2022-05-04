import typing as tp
from abc import ABC, abstractmethod
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import jax_metrics as jm
import optax
import treeo as to
import treex as tx

import elegy as eg
import elegy.modules.module as cm
from elegy import types, utils

M = tp.TypeVar("M", bound="ManagedModule")
A = tp.TypeVar("A")
ME = tp.TypeVar("ME", bound="jm.Metric")

InitStep = tp.Callable[[M, jnp.ndarray, types.Batch], M]
ResetStep = tp.Callable[[M], M]
PredStep = tp.Callable[[M, types.Batch, jnp.ndarray], tp.Tuple[types.Outputs, M]]
TestStep = tp.Callable[[M, types.Batch, jnp.ndarray], tp.Tuple[types.Logs, M]]
TrainStep = tp.Callable[
    [M, types.Batch, jnp.ndarray, jnp.ndarray],
    tp.Tuple[types.Logs, M],
]


class ManagedModule(cm.Module):
    # nodes
    key: tp.Optional[jnp.ndarray] = to.node()
    optimizer: tp.Optional[tx.Optimizer] = to.node()
    _logs: tp.Optional[tp.Dict[str, tp.Any]] = to.node()
    _avg_loss: jm.metrics.Mean = to.node()

    # statics
    strategy: "eg.Strategy"
    strategy_init_step: tp.Dict["eg.Strategy", InitStep["ManagedModule"]]
    strategy_reset_step: tp.Dict["eg.Strategy", ResetStep["ManagedModule"]]
    strategy_predict_step: tp.Dict["eg.Strategy", PredStep["ManagedModule"]]
    streategy_test_step: tp.Dict["eg.Strategy", TestStep["ManagedModule"]]
    strategy_train_step: tp.Dict["eg.Strategy", TrainStep["ManagedModule"]]

    def __init__(
        self,
        *,
        optimizer: tp.Optional[
            tp.Union[optax.GradientTransformation, tx.Optimizer]
        ] = None,
        initialized: bool = False,
        strategy: tp.Optional[tp.Union[str, "eg.Strategy"]] = None,
    ) -> None:
        from elegy.strategies import Eager, get_strategy

        super().__init__(initialized=initialized)

        self.key = None
        self.strategy = (
            get_strategy(strategy)
            if isinstance(strategy, str)
            else strategy
            if strategy is not None
            else Eager()
        )
        self.optimizer = (
            tx.Optimizer(optimizer)
            if isinstance(optimizer, optax.GradientTransformation)
            else optimizer
        )
        self._avg_loss = jm.metrics.Mean(name="loss")
        self._logs = None

        self.strategy_init_step = {}
        self.strategy_reset_step = {}
        self.strategy_predict_step = {}
        self.streategy_test_step = {}
        self.strategy_train_step = {}

        self.setup_strategy()

    def set_strategy(
        self: M,
        strategy: tp.Union[str, "eg.Strategy"],
    ) -> M:
        from elegy.strategies import get_strategy

        if isinstance(strategy, str):
            strategy = get_strategy(strategy)

        if self.strategy == strategy:
            return self

        if self.initialized:
            # current strategy to local
            self = self.strategy.to_local(self)
            # new strategy from local
            self = strategy.from_local(self)

        # update strategy
        self = self.replace(strategy=strategy)
        self.setup_strategy()

        return self

    def setup_strategy(self):
        strategy = self.strategy

        if strategy is not None and (
            strategy not in self.strategy_init_step
            or strategy not in self.strategy_reset_step
            or strategy not in self.strategy_predict_step
            or strategy not in self.streategy_test_step
            or strategy not in self.strategy_train_step
        ):
            # build strategy functions
            self.strategy_init_step[strategy] = strategy.init_step_fn(self)
            self.strategy_reset_step[strategy] = strategy.reset_step_fn(self)
            self.strategy_predict_step[strategy] = strategy.predict_step_fn(self)
            self.streategy_test_step[strategy] = strategy.test_step_fn(self)
            self.strategy_train_step[strategy] = strategy.train_step_fn(self)

    @property
    def init_step_fn(self) -> InitStep["ManagedModule"]:
        return self.strategy_init_step[self.strategy]

    @property
    def reset_step_fn(self) -> ResetStep["ManagedModule"]:
        return self.strategy_reset_step[self.strategy]

    @property
    def predict_step_fn(self) -> PredStep["ManagedModule"]:
        return self.strategy_predict_step[self.strategy]

    @property
    def test_step_fn(self) -> TestStep["ManagedModule"]:
        return self.streategy_test_step[self.strategy]

    @property
    def train_step_fn(self) -> TrainStep["ManagedModule"]:
        return self.strategy_train_step[self.strategy]

    # --------------------------------------------------------------------------
    # Managed API
    # --------------------------------------------------------------------------
    @abstractmethod
    def managed_init_step(
        self: M,
        key: jnp.ndarray,
        batch: tp.Any,
    ) -> M:
        ...

    # has default implematation
    def managed_reset_step(self: M) -> M:
        return self

    # optional
    def managed_predict_step(
        self: M,
        key: jnp.ndarray,
        batch: tp.Any,
        batch_idx: jnp.ndarray,
    ) -> tp.Tuple[types.Outputs, M]:
        raise types.MissingMethod()

    # optional
    def managed_test_step(
        self: M,
        key: jnp.ndarray,
        batch: tp.Any,
        batch_idx: jnp.ndarray,
    ) -> M:
        raise types.MissingMethod()

    # optional
    def managed_train_step(
        self: M,
        key: jnp.ndarray,
        batch: tp.Any,
        batch_idx: jnp.ndarray,
        epoch_idx: jnp.ndarray,
    ) -> tp.Tuple[types.Loss, M]:
        raise types.MissingMethod()

    @abstractmethod
    def get_params(self) -> tp.Any:
        ...

    @abstractmethod
    def set_params(self: M, params: tp.Any) -> M:
        ...

    @abstractmethod
    def get_batch_stats(self) -> tp.Any:
        ...

    @abstractmethod
    def set_batch_stats(self: M, batch_stats: tp.Any) -> M:
        ...

    # ---------------------------------------------------------------------------
    # core methods
    # ---------------------------------------------------------------------------

    def init_step(self: M, key: jnp.ndarray, batch: tp.Any) -> M:

        batch = self.strategy.lift_data(batch)
        key = self.strategy.lift_key(key)

        self = self.init_step_fn(self, key, batch)

        return self.replace(key=key)

    def _init_step_manager(self: M, key: jnp.ndarray, batch: tp.Any) -> M:

        init_key, key = jax.random.split(key)

        self = self.managed_init_step(init_key, batch)

        if self.optimizer is not None:
            optimizer = self.optimizer.init(self.get_params())
        else:
            optimizer = None

        self = self.replace(
            key=key,
            optimizer=optimizer,
            _avg_loss=self._avg_loss.init(),
        )

        # do this last
        self = self.strategy.handle_post_init(self)

        return self

    def reset_step(self: M) -> M:
        return self.reset_step_fn(self)

    def _reset_step_manager(self: M) -> M:
        self = self.managed_reset_step()

        metric_fields = {
            field: value.reset()
            for field, value in vars(self).items()
            if isinstance(value, jm.metrics.Metric)
        }

        return self.replace(**metric_fields)

    def predict_step(
        self: M, batch: tp.Any, batch_idx: int
    ) -> tp.Tuple[types.Outputs, M]:
        batch = self.strategy.lift_data(batch)

        outputs, self = self.predict_step_fn(self, batch, jnp.array(batch_idx))

        outputs = self.strategy.lower_outputs(outputs)

        return outputs, self

    def _predict_step_manager(
        self: M, batch: tp.Any, batch_idx: jnp.ndarray
    ) -> tp.Tuple[types.Outputs, M]:
        assert self.key is not None

        step_key, key = jax.random.split(self.key)

        outputs, self = self.managed_predict_step(step_key, batch, batch_idx)

        return outputs, self.replace(key=key)

    def test_step(self: M, batch: tp.Any, batch_idx: int) -> tp.Tuple[types.Logs, M]:
        batch = self.strategy.lift_data(batch)

        logs, self = self.test_step_fn(self, batch, jnp.array(batch_idx))

        return logs, self

    def _test_step_manager(
        self: M, batch: tp.Any, batch_idx: jnp.ndarray
    ) -> tp.Tuple[types.Logs, M]:
        assert self.key is not None
        step_key, key = jax.random.split(self.key)

        self = self.replace(_logs={})

        self = self.managed_test_step(step_key, batch, batch_idx)

        # get logs
        logs, self = self._process_logs()

        # verify _process_logs cleaned up _logs
        assert self._logs is None

        return logs, self.replace(key=key)

    def train_step(
        self: M, batch: tp.Any, batch_idx: int, epoch_idx: int
    ) -> tp.Tuple[types.Logs, M]:
        batch = self.strategy.lift_data(batch)

        logs, self = self.train_step_fn(
            self, batch, jnp.array(batch_idx), jnp.array(epoch_idx)
        )

        return logs, self

    def _loss_fn(
        self: M,
        params: tp.Any,
        key: jnp.ndarray,
        batch: tp.Any,
        batch_idx: jnp.ndarray,
        epoch_idx: jnp.ndarray,
    ) -> tp.Tuple[types.Loss, M]:
        self = self.set_params(params)
        loss, self = self.managed_train_step(key, batch, batch_idx, epoch_idx)
        return loss, self

    def _train_step_manager(
        self: M,
        batch: tp.Any,
        batch_idx: jnp.ndarray,
        epoch_idx: jnp.ndarray,
    ) -> tp.Tuple[types.Logs, M]:

        if self.optimizer is None:
            raise RuntimeError(
                "Trying to run `managed_train_step` but no `optimizer` was given"
            )

        assert self.key is not None
        step_key, key = jax.random.split(self.key)

        self = self.replace(_logs={})

        params = self.get_params()

        # caclulate gradients
        (loss, self), grads = jax.value_and_grad(self._loss_fn, has_aux=True)(
            params, step_key, batch, batch_idx, epoch_idx
        )

        # add average loss metric
        assert self._logs is not None
        self = self.log("_avg_loss", self._avg_loss.batch_updates(values=loss))

        # sync gradients
        grads = self.strategy.handle_grads(grads)

        # sync batch stats
        batch_stats = self.get_batch_stats()
        batch_stats = self.strategy.handle_batch_stats(batch_stats)
        self = self.set_batch_stats(batch_stats)

        # run optimizer and update params
        assert self.optimizer is not None
        params, optimizer = self.optimizer.update(grads, params)
        self = self.set_params(params)

        # get logs
        logs, self = self._process_logs()

        # verify _process_logs cleaned up _logs
        assert self._logs is None

        return logs, self.replace(
            key=key,
            optimizer=optimizer,
        )

    def _process_logs(self: M) -> tp.Tuple[types.Logs, M]:
        assert self._logs is not None

        log_updates = self._logs.copy()
        loss_updates: tp.Optional[jm.Metric] = log_updates.pop("_avg_loss", None)

        logs = {}
        names = set()
        field_updates: tp.Dict[str, jm.Metric] = {}

        def _update_metric(field: str, batch_updates: jm.Metric) -> types.Logs:
            metric: jm.Metric = getattr(self, field)
            batch_updates = self.strategy.handle_metrics(batch_updates)
            metric = metric.merge(batch_updates)
            metric_logs = metric.compute_logs()

            field_updates[field] = metric

            return metric_logs

        for name, value in log_updates.items():
            if isinstance(value, jnp.ndarray):
                metric_logs = {name: value}
            elif isinstance(value, jm.Metric):
                metric_logs = _update_metric(field=name, batch_updates=value)
            else:
                raise ValueError(
                    f"Unsupported metric type: {type(value)}, "
                    f"supported types are: ndarray, Metric"
                )

            for metric_name, metric_value in metric_logs.items():
                metric_name = utils.get_unique_name(names, metric_name)
                logs[metric_name] = metric_value

        if "loss" not in logs and loss_updates is not None:
            metric_logs = _update_metric("_avg_loss", loss_updates)
            logs["loss"] = metric_logs["loss"]

        return logs, self.replace(
            _logs=None,
            **field_updates,
        )

    # ------------------------------------------------------------------------
    # other methods
    # ------------------------------------------------------------------------
    def log(self: M, key: str, value: tp.Any) -> M:
        if self._logs is None:
            raise RuntimeError("Cannot use logs in this context.")

        if isinstance(value, jnp.ndarray):
            if value.size != 1:
                raise RuntimeError(
                    f"Array logs must be scalar, got '{value.shape}' for key '{key}'."
                )
        elif isinstance(value, jm.Metric):
            field = key
            batch_updates = value

            if not hasattr(self, field):
                raise ValueError(
                    f"Unknown field '{field}' in '{type(self).__name__}' when"
                    f"logging '{type(batch_updates).__name__}'. "
                    f"When logging batch updates to 'Metrics' make sure the "
                    f"key is the field where the metric is stored."
                )
            elif hasattr(self, field) and type(getattr(self, field)) != type(
                batch_updates
            ):
                field_value = getattr(self, field)
                raise ValueError(
                    f"Field '{field}' in '{type(self).__name__}' is of type "
                    f"'{type(field_value).__name__}' but trying to log "
                    f"metric of type '{type(batch_updates).__name__}'. "
                    f"When logging batch updates to 'Metrics' make sure the "
                    f"key is the field where the metric is stored."
                )

        logs = self._logs.copy()
        logs[key] = value

        return self.replace(_logs=logs)
