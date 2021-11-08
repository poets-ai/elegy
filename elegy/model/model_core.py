import enum
import pathlib
import typing as tp
from abc import ABC, abstractmethod
from dataclasses import dataclass

import cloudpickle
import einops
import jax
import jax.numpy as jnp
import numpy as np
import treex as tx
from jax.experimental import jax2tf

from elegy import types, utils

from . import utils as model_utils

try:
    import tensorflow as tf
except ImportError:
    tf = None

A = tp.TypeVar("A")
M = tp.TypeVar("M", bound="ModelCore")

PredStepOutput = tp.Tuple[tp.Any, M]
TestStepOutput = tp.Tuple[jnp.ndarray, types.Logs, M]
LossStepOutput = tp.Tuple[jnp.ndarray, tp.Tuple[types.Logs, M]]
GradStepOutput = tp.Tuple[M, types.Logs, M]
TrainStepOutput = tp.Tuple[types.Logs, M]

InitStep = tp.Callable[[M, jnp.ndarray, tp.Any], M]
PredStep = tp.Callable[[M, tp.Any], PredStepOutput[M]]
TestStep = tp.Callable[[M, tp.Any, types.Labels], TestStepOutput[M]]
GradStep = tp.Callable[[M, tp.Any, types.Labels], GradStepOutput[M]]
TrainStep = tp.Callable[[M, tp.Any, types.Labels], TrainStepOutput[M]]


# --------------------------------------------------------------------------------
# DistributedStrategy
# --------------------------------------------------------------------------------


class DistributedStrategy(ABC):
    def from_local(self, model: M) -> M:
        return model

    def to_local(self, model: M) -> M:
        return model

    def lift_data(self, data: A) -> A:
        return data

    def lift_key(self, key: jnp.ndarray) -> jnp.ndarray:
        return key

    def lift_batch_size(self, batch_size: int) -> int:
        return batch_size

    def handle_post_init(self, model: M) -> M:
        return model

    def handle_lm_kwargs(
        self,
        losses_kwargs: types.Logs,
        metrics_kwargs: types.Logs,
    ) -> tp.Tuple[types.Logs, types.Logs]:
        return losses_kwargs, metrics_kwargs

    def handle_lm_logs(
        self,
        losses_logs: types.Logs,
        metrics_logs: types.Logs,
    ) -> tp.Tuple[types.Logs, types.Logs]:
        return losses_logs, metrics_logs

    def handle_model_and_grads(
        self,
        model: M,
        grads: M,
    ) -> tp.Tuple[M, M]:
        return model, grads

    @abstractmethod
    def init_step_fn(self, model: M) -> InitStep[M]:
        ...

    @abstractmethod
    def pred_step_fn(self, model: M) -> PredStep[M]:
        ...

    @abstractmethod
    def test_step_fn(self, model: M) -> TestStep[M]:
        ...

    @abstractmethod
    def train_step_fn(self, model: M) -> TrainStep[M]:
        ...

    # implement order methods, required so that DistributedStrategy can be
    # used as a key in a dict
    def __lt__(self, other):
        return self.__class__.__name__ < other.__class__.__name__

    def __le__(self, other):
        return self.__class__.__name__ <= other.__class__.__name__

    def __gt__(self, other):
        return self.__class__.__name__ > other.__class__.__name__

    def __ge__(self, other):
        return self.__class__.__name__ >= other.__class__.__name__


@dataclass(unsafe_hash=True)
class Eager(DistributedStrategy):
    def init_step_fn(self, model: M) -> InitStep[M]:
        return model.__class__._static_init_step

    def pred_step_fn(self, model: M) -> PredStep[M]:
        return model.__class__._static_pred_step

    def test_step_fn(self, model: M) -> TestStep[M]:
        return model.__class__._static_test_step

    def train_step_fn(self, model: M) -> TrainStep[M]:
        return model.__class__._static_train_step


@dataclass(unsafe_hash=True)
class JIT(DistributedStrategy):
    def init_step_fn(self, model: M) -> InitStep[M]:
        return jax.jit(model.__class__._static_init_step)

    def pred_step_fn(self, model: M) -> PredStep[M]:
        return jax.jit(model.__class__._static_pred_step)

    def test_step_fn(self, model: M) -> TestStep[M]:
        return jax.jit(model.__class__._static_test_step)

    def train_step_fn(self, model: M) -> TrainStep[M]:
        return jax.jit(model.__class__._static_train_step)


@dataclass(unsafe_hash=True)
class DataParallel(DistributedStrategy):
    def from_local(self, model: M) -> M:
        # device_idxs used to inform pmap about the number of devices
        device_idxs = jnp.arange(jax.device_count())
        model = jax.pmap(
            lambda idx, model: model,
            in_axes=(0, None),
            out_axes=0,
        )(device_idxs, model)

        return model

    def to_local(self, model: M) -> M:
        return jax.tree_map(lambda x: x[0], model)

    def lift_data(self, data: A) -> A:
        data = jax.tree_map(
            lambda x: einops.rearrange(
                x,
                "(device batch) ... -> device batch ...",
                device=jax.device_count(),
            ),
            data,
        )
        return data

    def lift_key(self, key: jnp.ndarray) -> jnp.ndarray:
        key = einops.repeat(
            key,
            "... -> device ...",
            device=jax.device_count(),
        )
        return key

    def lift_batch_size(self, batch_size: int) -> int:
        return batch_size * jax.device_count()

    def handle_post_init(self, model: M) -> M:
        return model.map(
            lambda key: jax.random.fold_in(key, jax.lax.axis_index("device")),
            tx.Rng,
            inplace=True,
        )

    def handle_lm_kwargs(
        self,
        losses_kwargs: types.Logs,
        metrics_kwargs: types.Logs,
    ) -> tp.Tuple[types.Logs, types.Logs]:
        metrics_kwargs = jax.tree_map(
            lambda x: einops.rearrange(x, "device batch ... -> (device batch) ...")
            if x.ndim > 1
            else x,
            jax.lax.all_gather(metrics_kwargs, axis_name="device"),
        )

        return losses_kwargs, metrics_kwargs

    def handle_lm_logs(
        self,
        losses_logs: types.Logs,
        metrics_logs: types.Logs,
    ) -> tp.Tuple[types.Logs, types.Logs]:
        losses_logs = jax.lax.pmean(losses_logs, axis_name="device")

        return losses_logs, metrics_logs

    def handle_model_and_grads(
        self,
        model: M,
        grads: M,
    ) -> tp.Tuple[M, M]:
        grads = jax.lax.pmean(grads, axis_name="device")
        model = model.map(lambda x: jax.lax.pmean(x, axis_name="device"), tx.BatchStat)

        return model, grads

    def init_step_fn(self, model: M) -> InitStep[M]:
        return jax.pmap(
            model.__class__._static_init_step,
            axis_name="device",
        )

    def pred_step_fn(self, model: M) -> PredStep[M]:
        return jax.pmap(
            model.__class__._static_pred_step,
            axis_name="device",
        )

    def test_step_fn(self, model: M) -> TestStep[M]:
        return jax.pmap(
            model.__class__._static_test_step,
            axis_name="device",
            out_axes=(0, None, 0),  # None = logs not replicated
        )

    def train_step_fn(self, model: M) -> TrainStep[M]:
        return jax.pmap(
            model.__class__._static_train_step,
            axis_name="device",
            out_axes=(None, 0),  # None = logs not replicated
        )


# --------------------------------------------------------------------------------
# ModelCore
# --------------------------------------------------------------------------------


class ModelMeta(tx.TreeMeta):
    def __call__(self, *args, **kwargs) -> "ModelCore":
        model: ModelCore = super().__call__(*args, **kwargs)

        model.create_jit_functions()

        return model


class ModelCore(tx.Treex, tx.Filters, metaclass=ModelMeta):

    seed: tp.Union[int, jnp.ndarray] = 42
    _initialized: bool = False
    _distributed_strategy: DistributedStrategy = JIT()

    init_step_fn: tp.Dict[DistributedStrategy, InitStep]
    pred_step_fn: tp.Dict[DistributedStrategy, PredStep]
    test_step_fn: tp.Dict[DistributedStrategy, TestStep]
    train_step_fn: tp.Dict[DistributedStrategy, TrainStep]

    def __init__(
        self,
        eager: bool = False,
        seed: int = 42,
    ):
        self.seed = seed

        self._distributed_strategy = Eager() if eager else JIT()

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def is_distributed(self) -> bool:
        return (
            self._distributed_strategy != JIT()
            and self._distributed_strategy != Eager()
        )

    @property
    def is_eager(self) -> bool:
        return self._distributed_strategy == Eager()

    @property
    def distributed_strategy(self) -> DistributedStrategy:
        return self._distributed_strategy

    def create_jit_functions(self):
        cls = self.__class__
        self._jitted_members: tp.Set[str] = set()

        self.init_step_fn = {}
        self.pred_step_fn = {}
        self.test_step_fn = {}
        self.train_step_fn = {}

        self._maybe_build_strategy()

        self._jitted_members |= {
            "init_step_fn",
            "pred_step_fn",
            "test_step_fn",
            "train_step_fn",
        }

    def _maybe_build_strategy(self):
        strategy = self._distributed_strategy

        if (
            strategy not in self.init_step_fn
            or strategy not in self.pred_step_fn
            or strategy not in self.test_step_fn
            or strategy not in self.train_step_fn
        ):
            # build strategy functions
            self.init_step_fn[strategy] = strategy.init_step_fn(self)
            self.pred_step_fn[strategy] = strategy.pred_step_fn(self)
            self.test_step_fn[strategy] = strategy.test_step_fn(self)
            self.train_step_fn[strategy] = strategy.train_step_fn(self)

    def __setstate__(self, d):
        self.__dict__ = d
        self.create_jit_functions()

    def __getstate__(self):
        d = self.__dict__.copy()

        # remove jitted functions
        for member in self._jitted_members:
            if member in d:
                del d[member]

        return d

    def _update_from(self: M, other: M):
        self.__dict__.update(other.__dict__)

    # ----------------------------------------------------------------
    # Abstract API
    # ----------------------------------------------------------------

    def init_step(
        self: M,
        key: jnp.ndarray,
        inputs: tp.Any,
    ) -> M:
        raise types.MissingMethod()

    def pred_step(
        self: M,
        inputs: tp.Any,
    ) -> PredStepOutput[M]:
        raise types.MissingMethod()

    def test_step(
        self: M,
        inputs: tp.Any,
        labels: tp.Mapping[str, tp.Any],
    ) -> TestStepOutput[M]:
        raise types.MissingMethod()

    def grad_step(
        self: M,
        inputs: tp.Any,
        labels: tp.Mapping[str, tp.Any],
    ) -> GradStepOutput[M]:
        raise types.MissingMethod()

    def train_step(
        self: M,
        inputs: tp.Any,
        labels: tp.Mapping[str, tp.Any],
    ) -> TrainStepOutput[M]:
        raise types.MissingMethod()

    def reset_metrics(self) -> None:
        raise types.MissingMethod()

    # ----------------------------------------------------------------
    # static version
    # ----------------------------------------------------------------

    @staticmethod
    def _static_init_step(
        model: M,
        key: jnp.ndarray,
        inputs: tp.Any,
    ) -> M:
        return model.init_step(key, inputs)

    @staticmethod
    def _static_pred_step(
        model: M,
        inputs: tp.Any,
    ) -> PredStepOutput[M]:
        return model.pred_step(inputs)

    @staticmethod
    def _static_test_step(
        model: M,
        inputs: tp.Any,
        labels: tp.Mapping[str, tp.Any],
    ) -> TestStepOutput[M]:
        return model.test_step(inputs, labels)

    @staticmethod
    def _static_train_step(
        model: M,
        inputs: tp.Any,
        labels: tp.Mapping[str, tp.Any],
    ) -> TrainStepOutput[M]:
        return model.train_step(inputs, labels)

    # ----------------------------------------------------------------
    # distributed API
    # ----------------------------------------------------------------

    def distributed(
        self: M,
        strategy: tp.Optional[DistributedStrategy] = None,
        *,
        inplace: bool = False,
    ) -> M:
        model: M

        if strategy is None:
            strategy = DataParallel()

        if not inplace:
            model = self.copy()
        else:
            model = self

        if strategy == model.distributed_strategy:
            return model

        model = self._distributed_strategy.to_local(model)
        model = strategy.from_local(model)

        model._distributed_strategy = strategy
        model._maybe_build_strategy()

        if inplace:
            self._update_from(model)
            model = self

        return model

    def local(self: M, inplace: bool = False) -> M:
        return self.distributed(JIT(), inplace=inplace)

    def eager(self: M, inplace: bool = False) -> M:
        return self.distributed(Eager(), inplace=inplace)

    # ----------------------------------------------------------------
    # high-level methods
    # ----------------------------------------------------------------

    def init_on_batch(self, inputs: tp.Any):

        key = tx.Key(self.seed)

        inputs = self._distributed_strategy.lift_data(inputs)
        key = self._distributed_strategy.lift_key(key)

        init_step_fn = self.init_step_fn[self._distributed_strategy]
        model = init_step_fn(self, key, inputs)

        if not isinstance(model, type(self)):
            raise ValueError(
                f"Model.init_step() must return an instance of {type(self)}."
            )

        self._update_from(model)
        self._initialized = True

    def predict_on_batch(self, inputs: tp.Any) -> tp.Any:
        """
        Returns predictions for a single batch of samples.

        Arguments:
            x: Input data. A Numpy/Jax array (or array-like), or possibly
                nested python structure of dict, list, tuple that contain
                arrays as leafs.

        Returns:
            Jax array(s) of predictions.

        Raises:
            ValueError: In case of mismatch between given number of inputs and
                expectations of the model.
        """
        if not self._initialized:
            self.init_on_batch(inputs)

        # do this after init_on_batch()
        inputs = self._distributed_strategy.lift_data(inputs)

        self.eval(inplace=True)

        pred_step_fn = self.pred_step_fn[self._distributed_strategy]
        y_pred, model = pred_step_fn(self, inputs)

        if not isinstance(model, type(self)):
            raise ValueError(
                f"Model.pred_step() must return an instance of {type(self)}."
            )

        self._update_from(model)

        return y_pred

    def test_on_batch(
        self,
        inputs: tp.Any,
        labels: tp.Any,
    ) -> types.Logs:
        """
        Test the model on a single batch of samples.

        Arguments:
            x: Input data. It could be:

                - A Numpy array (or array-like), or a list
                    of arrays (in case the model has multiple inputs).
                - A dict mapping input names to the corresponding arrays, if
                    the model has named inputs.
            y: Target data. Like the input data `x`, it could be either Numpy
                array(s) or Jax array(s).
            sample_weight: Optional array of the same length as x, containing
                weights to apply to the model's loss for each sample. In the case of
                temporal data, you can pass a 2D array with shape (samples,
                sequence_length), to apply a different weight to every timestep of
                every sample.

        Returns:
            A `logs` dictionary of containing the main `loss` as well as all
            other losses and metrics.
        Raises:
            ValueError: In case of invalid user-provided arguments.
        """
        if not self._initialized:
            self.init_on_batch(inputs)

        # do this after init_on_batch()
        inputs, labels = self._distributed_strategy.lift_data((inputs, labels))

        if not isinstance(labels, tp.Mapping):
            labels = dict(target=labels)

        self.eval(inplace=True)

        test_step_fn = self.test_step_fn[self._distributed_strategy]
        loss, logs, model = test_step_fn(
            self,
            inputs,
            labels,
        )

        if not isinstance(model, type(self)):
            raise ValueError(
                f"Model.test_step() must return an instance of {type(self)}."
            )

        self._update_from(model)

        return logs

    def train_on_batch(
        self,
        inputs: tp.Any,
        labels: tp.Any,
    ) -> types.Logs:
        """
        Runs a single gradient update on a single batch of data.

        Arguments:
            x: Input data. It could be:

                - A Numpy array (or array-like), or a iterable of arrays
                    (in case the model has multiple inputs).
                - A dict mapping input names to the corresponding arrays,
                    if the model has named inputs.
            y: Target data. Like the input data `x`, it could be either Numpy
                array(s) or Jax array(s). It should be consistent with `x`
                (you cannot have Numpy inputs and array targets, or inversely).
            sample_weight: Optional array of the same length as x, containing
                weights to apply to the model's loss for each sample. In the case of
                temporal data, you can pass a 2D array with shape (samples,
                sequence_length), to apply a different weight to every timestep of
                every sample.
            class_weight: Optional dictionary mapping class indices (integers) to a
                weight (float) to apply to the model's loss for the samples from this
                class during training. This can be useful to tell the model to "pay
                more attention" to samples from an under-represented class.

        Returns:
            A `logs` dictionary of containing the main `loss` as well as all
            other losses and metrics.

        Raises:
            ValueError: In case of invalid user-provided arguments.
        """

        if not self.initialized:
            self.init_on_batch(inputs)

        # do this after init_on_batch()
        inputs, labels = self._distributed_strategy.lift_data((inputs, labels))

        self.train(inplace=True)

        if not isinstance(labels, tp.Mapping):
            labels = dict(target=labels)

        train_step_fn = self.train_step_fn[self._distributed_strategy]
        logs, model = train_step_fn(self, inputs, labels)

        if not isinstance(model, type(self)):
            raise ValueError(
                f"Model.train_step() must return an instance of {type(self)}."
            )

        self._update_from(model)

        return logs

    def save(
        self,
        path: tp.Union[str, pathlib.Path],
    ) -> None:
        """
        Saves the model to disk.

        It creates a directory that includes:

        - `{path}/model.pkl`: The `Model` object instance serialized with `pickle`,
            this allows you to re-instantiate the model later.

        This allows you to save the entirety of the states of a model
        in a directory structure which can be fully restored via
        `Model.load` if the model is already instiated or `elegy.model.load`
        to load the model instance from its pickled version.

        ```python
        import elegy

        model.save('my_model')  # creates folder at 'my_model'
        del model  # deletes the existing model

        # returns a model identical to the previous one
        model = elegy.model.load('my_model')
        ```
        Arguments:
            path: path where model structure will be saved.
        """
        model = self.local()

        if isinstance(path, str):
            path = pathlib.Path(path)

        path.mkdir(parents=True, exist_ok=True)

        with open(path / "model.pkl", "wb") as f:
            cloudpickle.dump(model, f)

    def load(
        self,
        path: tp.Union[str, pathlib.Path],
    ) -> None:
        """
        Loads all weights + states from a folder structure.

        You can load states from other models that have slightly different architecture
        as long as long as it preserves the ordering of the `haiku.Params` + `haiku.State`
        structures, adding or removing layers is fine as long as they don't have weights,
        new layers with weights will be initialized from scratch.

        Arguments:
            path: path to a saved model's directory.
        """
        pass

    def saved_model(
        self,
        inputs: tp.Any,
        path: tp.Union[str, pathlib.Path],
        batch_size: tp.Union[int, tp.Sequence[int], None] = None,
    ):
        """
        Serializes the prediction function of the Model (`pred_step`) as a TensorFlow SavedModel via
        `jax2tf`.

        !!! Note
            Due to a current limitation in JAX it is not possible to create dynamicly
            shaped SavedModels so you must specify the `batch_size` argument to create
            one or more statically shaped versions / signatures: [jax#5915](https://github.com/google/jax/issues/5915).

        Arguments:
            x: A sample input used to infer shapes.
            path: The path where the SavedModel should be saved.
            batch_size: An integer or sequence of integers specifying the size of the batch
                dimension of each of the resulting SavedModel's signatures.

        """
        model = self.local()

        if not model.initialized:
            model.init_on_batch(inputs)

        if model_utils.convert_and_save_model is None:
            raise ImportError(f"Could not import tensorflow.")

        if isinstance(batch_size, int):
            batch_size = [batch_size]

        if isinstance(path, str):
            path = pathlib.Path(path)

        path.mkdir(parents=True, exist_ok=True)

        inputs = jax.tree_map(jnp.asarray, inputs)

        if batch_size is None:
            input_signatures = [
                jax.tree_map(
                    lambda p: tf.TensorSpec(shape=(None,) + p.shape[1:], dtype=p.dtype),
                    inputs,
                )
            ]
            shape_polymorphic_input_spec = jax.tree_map(
                lambda p: jax2tf.shape_poly.PolyShape(
                    "batch", *(["_"] * (len(p.shape) - 1))
                ),
                inputs,
            )
        else:
            input_signatures = [
                jax.tree_map(
                    lambda p: tf.TensorSpec(
                        shape=(batch_size,) + p.shape[1:], dtype=p.dtype
                    ),
                    inputs,
                )
                for batch_size in batch_size
            ]
            shape_polymorphic_input_spec = None

        flat_states, states_def = jax.tree_flatten(model)

        def jax_fn(flat_states, inputs):
            model: ModelCore = jax.tree_unflatten(states_def, flat_states)

            y_pred, _ = model.pred_step(
                inputs=inputs,
            )

            return y_pred

        model_utils.convert_and_save_model(
            jax_fn,
            flat_states,
            str(path),
            input_signatures=input_signatures,
            shape_polymorphic_input_spec=shape_polymorphic_input_spec,
            with_gradient=False,
            enable_xla=True,
            compile_model=False,
            save_model_options=None,
        )
