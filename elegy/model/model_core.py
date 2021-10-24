from abc import abstractmethod
import pathlib
import typing as tp

import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np
import treeo as to
from treeo.utils import Opaque
import treex as tx
from elegy import types, utils
from jax._src.numpy.lax_numpy import ndarray

from . import utils as model_utils

try:
    import tensorflow as tf
except ImportError:
    tf = None

M = tp.TypeVar("M", bound="ModelCore")

PredStep = tp.Tuple[tp.Any, M]
TestStep = tp.Tuple[types.Scalar, types.Logs, M]
GradStep = tp.Tuple[
    types.Scalar,
    types.Logs,
    types.Grads,
    M,
]


TrainStep = tp.Tuple[types.Logs, M]


class ModelMeta(to.TreeMeta):
    def __call__(self, *args, **kwargs) -> "ModelCore":
        model: ModelCore = super().__call__(*args, **kwargs)

        model.jit_step()

        return model


class ModelCore(tx.Treex, metaclass=ModelMeta):

    seed: tp.Union[int, jnp.ndarray] = 42
    eager: bool = False
    _initialized: bool = tx.static(default=False)

    def __init__(
        self,
        eager: bool = False,
        seed: tp.Union[int, jnp.ndarray] = 42,
    ):
        self.eager = eager
        self.seed = seed if isinstance(seed, jnp.ndarray) else jax.random.PRNGKey(seed)

    @property
    def initialized(self) -> bool:
        return self._initialized

    def jit_step(self):
        cls = self.__class__
        self._jitted_members: tp.Set[str] = set()

        self.init_step_jit = jax.jit(cls.init_step)
        self.pred_step_jit = jax.jit(cls.pred_step)
        self.test_step_jit = jax.jit(cls.test_step)
        self.train_step_jit = jax.jit(cls.train_step)

        self._jitted_members |= {
            "init_step_jit",
            "pred_step_jit",
            "test_step_jit",
            "train_step_jit",
        }

    def __setstate__(self, d):
        self.__dict__ = d
        self.jit_step()

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
    ) -> M:
        raise types.MissingMethod()

    def pred_step(
        self: M,
        inputs: tp.Any,
    ) -> PredStep[M]:
        raise types.MissingMethod()

    def test_step(
        self: M,
        inputs: tp.Any,
        labels: tp.Mapping[str, tp.Any],
    ) -> TestStep[M]:
        raise types.MissingMethod()

    def grad_step(
        self: M,
        inputs: tp.Any,
        labels: tp.Mapping[str, tp.Any],
    ) -> GradStep[M]:
        raise types.MissingMethod()

    def train_step(
        self: M,
        inputs: tp.Any,
        labels: tp.Mapping[str, tp.Any],
    ) -> TrainStep[M]:
        raise types.MissingMethod()

    def reset_metrics(self) -> None:
        raise types.MissingMethod()

    # ----------------------------------------------------------------
    # high-level methods
    # ----------------------------------------------------------------

    def init_on_batch(self):
        key = tx.Key(self.seed)

        if self.eager:
            model = self.init_step(key)
        else:
            model = self.init_step_jit(self, key)

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
            self.init_on_batch()

        self.eval(inplace=True)

        if self.eager:
            y_pred, model = self.pred_step(inputs)
        else:
            y_pred, model = self.pred_step_jit(self, inputs)

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
            self.init_on_batch()

        self.eval(inplace=True)

        if self.eager:
            loss, logs, model = self.test_step(
                inputs,
                labels,
            )
        else:
            loss, logs, model = self.test_step_jit(
                self,
                inputs,
                labels,
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
        if not self._initialized:
            self.init_on_batch()

        if not isinstance(labels, tp.Mapping):
            labels = dict(target=labels)

        if self.eager:
            logs, model = self.train_step(inputs, labels)
        else:
            logs, model = self.train_step_jit(self, inputs, labels)

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
            compile_model=True,
            save_model_options=None,
        )
