import typing as tp

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax_metrics as jm
import optax
import treeo as to
import treex as tx
from attr import mutable
from flax.core.frozen_dict import FrozenDict
from flax.core.scope import CollectionFilter, DenyList

import elegy as eg
import elegy.pytree as pytree_m
from elegy import types, utils
from elegy.modules.high_level.high_level_module import HighLevelModule

M = tp.TypeVar("M", bound="FlaxModule")
FrozerVariables = FrozenDict[str, tp.Mapping[str, tp.Any]]
Variables = tp.Mapping[str, tp.Mapping[str, tp.Any]]


class FlaxMixin:
    variables: tp.Optional[FrozerVariables]

    def get_params(self) -> tp.Optional[FrozerVariables]:

        return self.variables["params"] if "params" in self.variables else None

    def set_params(self: M, params: tp.Optional[FrozerVariables]) -> M:

        if params is not None:
            return self.replace(
                variables=self.variables.copy({"params": params}),
            )
        else:
            return self

    def get_batch_stats(self) -> tp.Optional[FrozerVariables]:

        return (
            self.variables["batch_stats"] if "batch_stats" in self.variables else None
        )

    def set_batch_stats(self: M, batch_stats: tp.Optional[FrozerVariables]) -> M:

        if batch_stats is not None:
            return self.replace(
                variables=self.variables.copy({"batch_stats": batch_stats}),
            )
        else:
            return self


class FlaxModule(FlaxMixin, HighLevelModule):
    # nodes
    variables: FrozerVariables
    # static
    _module: to.Hashable[nn.module.Module] = pytree_m.static_field()
    mutable_init: CollectionFilter = pytree_m.static_field()
    mutable_train: CollectionFilter = pytree_m.static_field()
    mutable_eval: CollectionFilter = pytree_m.static_field()
    rngs_init: tp.Tuple[str, ...] = pytree_m.static_field()
    rngs_train: tp.Tuple[str, ...] = pytree_m.static_field()
    rngs_eval: tp.Tuple[str, ...] = pytree_m.static_field()
    method_init: tp.Callable[..., tp.Any] = pytree_m.static_field()
    method_train: tp.Callable[..., tp.Any] = pytree_m.static_field()
    method_eval: tp.Callable[..., tp.Any] = pytree_m.static_field()
    init_training_value: bool = pytree_m.static_field()
    collection_keep: tp.Tuple[str, ...] = pytree_m.static_field()
    logs_full_path: bool = pytree_m.static_field()

    def __init__(
        self,
        module: nn.module.Module,
        *,
        # nodes
        variables: tp.Optional[Variables] = None,
        # static
        mutable_init: CollectionFilter = True,
        mutable_train: CollectionFilter = DenyList(["params"]),
        mutable_eval: CollectionFilter = DenyList(["params", "batch_stats"]),
        rngs_init: tp.Sequence[str] = ("params", "dropout", "any"),
        rngs_train: tp.Sequence[str] = ("dropout", "any"),
        rngs_eval: tp.Sequence[str] = ("any",),
        method_init: tp.Union[str, tp.Callable[..., tp.Any]] = "__call__",
        method_train: tp.Union[str, tp.Callable[..., tp.Any]] = "__call__",
        method_eval: tp.Union[str, tp.Callable[..., tp.Any]] = "__call__",
        init_training_value: bool = True,
        collection_keep: tp.Sequence[str] = ("params", "batch_stats", "cache"),
        logs_full_path: bool = False,
        # super
        losses_and_metrics: tp.Optional[jm.LossesAndMetrics] = None,
        optimizer: tp.Optional[
            tp.Union[optax.GradientTransformation, tx.Optimizer]
        ] = None,
        initialized: bool = False,
        strategy: tp.Optional[tp.Union[str, "eg.Strategy"]] = None,
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            initialized=initialized,
            strategy=strategy,
            losses_and_metrics=losses_and_metrics,
        )
        # module
        self._module = tx.Hashable(module)
        # node
        self.variables = (
            FrozenDict(variables) if variables is not None else FrozenDict()
        )
        # static
        self.mutable_init = mutable_init
        self.mutable_train = mutable_train
        self.mutable_eval = mutable_eval
        self.rngs_init = tuple(rngs_init)
        self.rngs_train = tuple(rngs_train)
        self.rngs_eval = tuple(rngs_eval)
        self.method_init = _unbounded_method(module, method_init)
        self.method_train = _unbounded_method(module, method_train)
        self.method_eval = _unbounded_method(module, method_eval)
        self.init_training_value = init_training_value
        self.collection_keep = tuple(collection_keep)
        self.logs_full_path = logs_full_path

    @property
    def module(self) -> nn.module.Module:
        return self._module.value

    # ---------------------------------------------------------------------------------
    # HighLevel API helpers
    # ---------------------------------------------------------------------------------

    def get_aux_losses(self) -> types.Logs:

        if "losses" in self.variables:
            losses = utils.flatten_names_unique(
                self.variables["losses"], only_last=not self.logs_full_path
            )
            return losses
        else:
            return {}

    def get_aux_metrics(self) -> types.Logs:

        if "metrics" in self.variables:
            metrics = utils.flatten_names_unique(
                self.variables["metrics"], only_last=not self.logs_full_path
            )
            return metrics
        else:
            return {}

    # ---------------------------------------------------------------------------------
    # HighLevel API init/apply
    # ---------------------------------------------------------------------------------

    def init(self: M, key: jnp.ndarray, inputs: tp.Any) -> M:
        outputs, self = self.apply(
            key,
            inputs,
            training=self.init_training_value,
            _initializing=True,
        )
        return self

    def apply(
        self: M,
        key: jnp.ndarray,
        inputs: tp.Any,
        training: bool,
        _initializing: bool = False,
    ) -> tp.Tuple[types.Outputs, M]:

        method = (
            self.method_init
            if _initializing
            else self.method_train
            if training
            else self.method_eval
        )
        mutable = (
            self.mutable_init
            if _initializing
            else self.mutable_train
            if training
            else self.mutable_eval
        )
        rng_names = (
            self.rngs_init
            if _initializing
            else self.rngs_train
            if training
            else self.rngs_eval
        )

        arg_names = utils._function_argument_names(method)
        args, kwargs = utils._split_args_kwargs(inputs)

        if (
            arg_names is not None
            and "training" in arg_names
            and "training" not in kwargs
        ):
            kwargs["training"] = training

        rngs = _split_into_collection(key, rng_names)

        apply_output = self.module.apply(
            self.variables,
            *args,
            rngs=rngs,
            mutable=mutable,
            method=method,
            **kwargs,
        )

        if mutable is False:
            outputs = apply_output
            variables = self.variables
        else:
            outputs, variable_updates = apply_output
            variables = self.variables.copy(variable_updates)

        return outputs, self.replace(variables=variables)

    # ---------------------------------------------------------------------------------
    # filter variables overrides
    # ---------------------------------------------------------------------------------

    def managed_init_step(self: M, key: jnp.ndarray, batch: tp.Any) -> M:
        self = super().managed_init_step(key, batch)
        return self._filter_variables()

    def managed_train_step(
        self: M,
        key: jnp.ndarray,
        batch: tp.Any,
        batch_idx: jnp.ndarray,
        epoch_idx: jnp.ndarray,
    ) -> tp.Tuple[types.Loss, M]:
        loss, self = super().managed_train_step(key, batch, batch_idx, epoch_idx)
        return loss, self._filter_variables()

    def managed_test_step(
        self: M, key: jnp.ndarray, batch: tp.Any, batch_idx: jnp.ndarray
    ) -> M:
        self = super().managed_test_step(key, batch, batch_idx)
        return self._filter_variables()

    def managed_predict_step(
        self: M, key: jnp.ndarray, batch: tp.Any, batch_idx: jnp.ndarray
    ) -> tp.Tuple[types.Outputs, M]:
        outputs, self = super().managed_predict_step(key, batch, batch_idx)
        return outputs, self._filter_variables()

    # ---------------------------------------------------------------------------
    # utility methods
    # ---------------------------------------------------------------------------

    def _filter_variables(self: M) -> M:

        variables = FrozenDict(
            {
                collection: self.variables[collection]
                for collection in self.collection_keep
                if collection in self.variables
            }
        )

        return self.replace(variables=variables)


def _unbounded_method(
    module: nn.module.Module,
    method: tp.Union[str, tp.Callable[..., tp.Any]],
) -> tp.Callable[..., tp.Any]:
    if isinstance(method, str):
        return getattr(module.__class__, method)
    return method


def _split_into_collection(
    key: jnp.ndarray,
    collection: tp.Sequence[str],
) -> tp.Dict[str, jnp.ndarray]:
    """
    Split the key into the specified rngs.
    """

    keys = jax.random.split(key, len(collection))

    keys_collection = {col: keys[i] for i, col in enumerate(collection)}

    return keys_collection
