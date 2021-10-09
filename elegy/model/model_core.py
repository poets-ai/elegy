import pathlib
import typing as tp

import cloudpickle
import jax
from jax._src.numpy.lax_numpy import ndarray
import jax.numpy as jnp
import numpy as np
import treex as tx
from elegy import types, utils

from . import utils as model_utils

try:
    import tensorflow as tf
except ImportError:
    tf = None

T = tp.TypeVar("T", bound="ModelCore")

PredStep = tp.Tuple[tp.Any, T]

TestStep = tp.Tuple[types.Scalar, types.Logs, T]

GradStep = tp.Tuple[
    types.Scalar,
    types.Logs,
    types.Grads,
    T,
]


TrainStep = tp.Tuple[types.Logs, T]


class ModelCore(tx.Module):

    history: tp.Dict[str, tp.Any]
    run_eagerly: bool = False
    sample_x: tp.Optional[tp.Any]
    seed: jnp.ndarray

    def __init__(
        self,
        run_eagerly: bool = False,
        sample_x: tp.Optional[tp.Any] = None,
        seed: tp.Union[int, jnp.ndarray] = 42,
    ):

        self.history = {}
        self.run_eagerly = run_eagerly
        self.sample_x = sample_x
        self.seed = seed if isinstance(seed, jnp.ndarray) else jax.random.PRNGKey(seed)

        self.jitted_members: tp.Set[str] = set()

        self.jit_step()

    def jit_step(self):
        self.call_init_step_jit = jax.jit(
            self.__class__.call_init_step,
            static_argnums=[],
        )
        self.call_pred_step_jit = jax.jit(
            self.__class__.call_pred_step,
            static_argnums=[],
        )
        self.call_test_step_jit = jax.jit(
            self.__class__.call_test_step,
            static_argnums=[],
        )
        self.call_train_step_jit = jax.jit(
            self.__class__.call_train_step,
            static_argnums=[],
        )

        self.jitted_members |= {
            "call_init_step_jit",
            "call_pred_step_jit",
            "call_test_step_jit",
            "call_train_step_jit",
        }

    def __setstate__(self, d):
        self.__dict__ = d
        self.jit_step()

    def __getstate__(self):
        d = self.__dict__.copy()

        # remove jitted functions
        for member in self.jitted_members:
            if member in d:
                del d[member]

        return d

    # ----------------------------------------------------------------
    # Abstract API
    # ----------------------------------------------------------------

    def update_modules(self):
        pass

    def init_step(
        self,
        key: jnp.ndarray,
    ) -> "ModelCore":
        return self.init(key)

    def call_init_step(
        self,
        key: jnp.ndarray,
    ) -> "ModelCore":
        return self.init_step(key)

    def pred_step(
        self: T,
        x: tp.Any,
    ) -> PredStep[T]:
        raise types.MissingMethod()

    def call_pred_step(
        self: T,
        x: tp.Any,
    ) -> PredStep[T]:
        return utils.inject_dependencies(self.pred_step)(
            x=x,
        )

    def test_step(
        self: T,
        x: tp.Any,
        y_true: tp.Any,
        sample_weight: tp.Optional[np.ndarray],
        class_weight: tp.Optional[np.ndarray],
    ) -> TestStep[T]:
        raise types.MissingMethod()

    def call_test_step(
        self: T,
        x: tp.Any,
        y_true: tp.Any,
        sample_weight: tp.Optional[np.ndarray],
        class_weight: tp.Optional[np.ndarray],
    ) -> TestStep[T]:
        return utils.inject_dependencies(self.test_step)(
            x=x,
            y_true=y_true,
            sample_weight=sample_weight,
            class_weight=class_weight,
        )

    def grad_step(
        self: T,
        x: tp.Any,
        y_true: tp.Any,
        sample_weight: tp.Optional[np.ndarray],
        class_weight: tp.Optional[np.ndarray],
        training: bool,
    ) -> GradStep[T]:
        raise types.MissingMethod()

    def train_step(
        self: T,
        x: tp.Any,
        y_true: tp.Any,
        sample_weight: tp.Optional[np.ndarray],
        class_weight: tp.Optional[np.ndarray],
    ) -> TrainStep[T]:
        raise types.MissingMethod()

    def call_train_step(
        self: T,
        x: tp.Any,
        y_true: tp.Any,
        sample_weight: tp.Optional[np.ndarray],
        class_weight: tp.Optional[np.ndarray],
    ) -> TrainStep[T]:
        return utils.inject_dependencies(self.train_step)(
            x=x,
            y_true=y_true,
            sample_weight=sample_weight,
            class_weight=class_weight,
        )

    # ----------------------------------------------------------------
    # high-level methods
    # ----------------------------------------------------------------

    def init_on_batch(
        self,
        key: jnp.ndarray,
    ):
        if self.run_eagerly:
            model = self.init_step(key)
        else:
            model = self.call_init_step_jit(self, key)

        self.update(model, inplace=True)

        # update self.initialized
        def initialize_inplace(module):
            if isinstance(module, tx.Module):
                module._initialized = True
            return module

        tx.module_map(initialize_inplace, self, inplace=True)

    def maybe_init_on_batch(
        self,
        x: tp.Optional[tp.Any],
    ):
        if not self.initialized:
            old_sample_x = self.sample_x
            self.sample_x = x

            try:
                self.init_on_batch(self.seed)
            finally:
                self.sample_x = old_sample_x

    def predict_on_batch(self, x: tp.Any) -> tp.Any:
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
        self.maybe_init_on_batch(x)

        self.eval(inplace=True)

        if self.run_eagerly:
            y_pred, model = self.call_pred_step(x)
        else:
            y_pred, model = self.call_pred_step_jit(self, x)

        self.update(model, inplace=True)

        return y_pred

    def test_on_batch(
        self,
        x: tp.Any,
        y: tp.Union[np.ndarray, tp.Mapping[str, tp.Any], tp.Tuple, None] = None,
        sample_weight: tp.Optional[np.ndarray] = None,
        class_weight: tp.Optional[np.ndarray] = None,
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
        self.maybe_init_on_batch(x)

        self.eval(inplace=True)

        if self.run_eagerly:
            loss, logs, model = self.call_test_step(
                x,
                y,
                sample_weight,
                class_weight,
            )
        else:
            loss, logs, model = self.call_test_step_jit(
                self,
                x,
                y,
                sample_weight,
                class_weight,
            )

        self.update(model, inplace=True)

        return logs

    def train_on_batch(
        self,
        x: tp.Any,
        y: tp.Union[np.ndarray, tp.Mapping[str, tp.Any], tp.Tuple, None] = None,
        sample_weight: tp.Optional[np.ndarray] = None,
        class_weight: tp.Optional[tp.Any] = None,
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
        self.maybe_init_on_batch(x)

        if self.run_eagerly:
            logs, model = self.call_train_step(
                x,
                y,
                sample_weight,
                class_weight,
            )
        else:
            logs, model = self.call_train_step_jit(
                self,
                x,
                y,
                sample_weight,
                class_weight,
            )

        self.update(model, inplace=True)

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
        if isinstance(path, str):
            path = pathlib.Path(path)

        path.mkdir(parents=True, exist_ok=True)

        with open(path / "model.pkl", "wb") as f:
            cloudpickle.dump(self, f)

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
        raise NotImplementedError()
        if isinstance(path, str):
            path = pathlib.Path(path)

    def saved_model(
        self,
        x: types.Pytree,
        path: tp.Union[str, pathlib.Path],
        batch_size: tp.Union[int, tp.Sequence[int]],
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

        if not self.initialized:
            raise types.ModelNotInitialized(
                f"Model not initialized, please execute `init` or `init_on_batch` before running this method."
            )

        if model_utils.convert_and_save_model is None:
            raise ImportError(f"Could not import tensorflow.")

        if isinstance(batch_size, int):
            batch_size = [batch_size]

        if isinstance(path, str):
            path = pathlib.Path(path)

        path.mkdir(parents=True, exist_ok=True)

        x = jax.tree_map(jnp.asarray, x)

        # polymorphic batch size currently not supported by jax: https://github.com/google/jax/issues/5915
        # -----------------------------------------
        # if batch_size is None:
        #     input_signatures = [
        #         jax.tree_map(
        #             lambda p: tf.TensorSpec(shape=(None,) + p.shape[1:], dtype=p.dtype),
        #             x,
        #         )
        #     ]
        #     shape_polymorphic_input_spec = jax.tree_map(
        #         lambda p: "(" + ", ".join(["batch"] + ["_"] * (len(p.shape) - 1)) + ")",
        #         x,
        #     )
        # else:
        input_signatures = [
            jax.tree_map(
                lambda p: tf.TensorSpec(
                    shape=(batch_size,) + p.shape[1:], dtype=p.dtype
                ),
                x,
            )
            for batch_size in batch_size
        ]
        shape_polymorphic_input_spec = None

        flat_states, states_def = jax.tree_flatten(self)

        def jax_fn(flat_states, inputs):
            model: ModelCore = jax.tree_unflatten(states_def, flat_states)

            y_pred, _ = utils.inject_dependencies(model.pred_step)(
                x=inputs, training=False
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
            compile_model=True,
            save_model_options=None,
        )

    def reset_metrics(self):
        raise NotImplementedError()
