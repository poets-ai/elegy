import pathlib
import pickle
import threading
import typing as tp
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass
from enum import Enum
from io import StringIO

import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np
import optax
import toolz
import yaml
from elegy import hooks, module, types, utils
from elegy.losses.loss import Loss
from elegy.metrics.metric import Metric
from tabulate import tabulate

from . import utils as model_utils

try:
    import tensorflow as tf
except ImportError:
    tf = None


class PredStep(tp.NamedTuple):
    y_pred: tp.Any
    states: types.States


class TestStep(tp.NamedTuple):
    loss: types.Scalar
    logs: types.Logs
    states: types.States


class GradStep(tp.NamedTuple):
    loss: types.Scalar
    logs: types.Logs
    states: types.States
    grads: types.Grads


class TrainStep(tp.NamedTuple):
    logs: types.Logs
    states: types.States


class ModelCore:
    states: types.States
    initial_states: types.States
    history: tp.Dict[str, tp.Any]
    run_eagerly: bool = False
    initialized: bool = False

    def __init__(
        self,
        states: tp.Optional[types.States] = None,
        run_eagerly: bool = False,
        initialized: bool = False,
    ):

        if states is None:
            states = types.States()

        self.initial_states = states
        self.states = states.copy()  # explicity do this to copy RNGSeq
        self.run_eagerly = run_eagerly
        self.history = {}
        self.initialized = initialized
        self.jitted_members: tp.Set[str] = set()

        self.jit_step()

    def jit_step(self):
        self.call_pred_step_jit = jax.jit(
            self.call_pred_step,
            static_argnums=[2, 3],
        )
        self.call_test_step_jit = jax.jit(
            self.call_test_step,
            static_argnums=[5, 6],
        )
        self.call_train_step_jit = jax.jit(
            self.call_train_step,
            static_argnums=[5, 6],
        )
        self.call_init_step_jit = jax.jit(
            self.call_init_step,
            static_argnums=[],
        )

        self.jitted_members |= {
            "call_pred_step_jit",
            "call_test_step_jit",
            "call_train_step_jit",
            "call_init_step_jit",
        }

    def __setstate__(self, d):
        self.__dict__ = d
        self.jit_step()

    def __getstate__(self):
        d = self.__dict__.copy()

        # remove states
        del d["states"]
        del d["initial_states"]

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

    def states_step(self) -> types.States:
        return types.States()

    def init_step(
        self,
        x: tp.Any,
        y_true: tp.Any,
        sample_weight: tp.Optional[np.ndarray],
        class_weight: tp.Optional[np.ndarray],
        states: types.States,
    ) -> types.States:
        raise types.MissingMethod()

    def call_init_step(
        self,
        x: tp.Any,
        y_true: tp.Any,
        sample_weight: tp.Optional[np.ndarray],
        class_weight: tp.Optional[np.ndarray],
        states: types.States,
    ) -> types.States:
        return utils.inject_dependencies(self.init_step)(
            x=x,
            y_true=y_true,
            sample_weight=sample_weight,
            class_weight=class_weight,
            states=states,
        )

    def pred_step(
        self,
        x: tp.Any,
        states: types.States,
        initializing: bool,
        training: bool,
    ) -> PredStep:
        raise types.MissingMethod()

    def call_pred_step(
        self,
        x: tp.Any,
        states: types.States,
        initializing: bool,
        training: bool,
    ) -> PredStep:
        return utils.inject_dependencies(self.pred_step)(
            x=x,
            states=states,
            initializing=initializing,
            training=training,
        )

    def test_step(
        self,
        x: tp.Any,
        y_true: tp.Any,
        sample_weight: tp.Optional[np.ndarray],
        class_weight: tp.Optional[np.ndarray],
        states: types.States,
        initializing: bool,
        training: bool,
    ) -> TestStep:
        raise types.MissingMethod()

    def call_test_step(
        self,
        x: tp.Any,
        y_true: tp.Any,
        sample_weight: tp.Optional[np.ndarray],
        class_weight: tp.Optional[np.ndarray],
        states: types.States,
        initializing: bool,
        training: bool,
    ) -> TestStep:
        return utils.inject_dependencies(self.test_step)(
            x=x,
            y_true=y_true,
            sample_weight=sample_weight,
            class_weight=class_weight,
            states=states,
            initializing=initializing,
            training=training,
        )

    def grad_step(
        self,
        x: tp.Any,
        y_true: tp.Any,
        sample_weight: tp.Optional[np.ndarray],
        class_weight: tp.Optional[np.ndarray],
        states: types.States,
        initializing: bool,
        training: bool,
    ) -> GradStep:
        raise types.MissingMethod()

    def train_step(
        self,
        x: tp.Any,
        y_true: tp.Any,
        sample_weight: tp.Optional[np.ndarray],
        class_weight: tp.Optional[np.ndarray],
        states: types.States,
        initializing: bool,
        training: bool,
    ) -> TrainStep:
        raise types.MissingMethod()

    def call_train_step(
        self,
        x: tp.Any,
        y_true: tp.Any,
        sample_weight: tp.Optional[np.ndarray],
        class_weight: tp.Optional[np.ndarray],
        states: types.States,
        initializing: bool,
        training: bool,
    ) -> TrainStep:
        return utils.inject_dependencies(self.train_step)(
            x=x,
            y_true=y_true,
            sample_weight=sample_weight,
            class_weight=class_weight,
            states=states,
            initializing=initializing,
            training=training,
        )

    # ----------------------------------------------------------------
    # high-level methods
    # ----------------------------------------------------------------

    def init_on_batch(
        self,
        x: tp.Any = (),
        y_true: tp.Any = None,
        sample_weight: tp.Optional[np.ndarray] = None,
        class_weight: tp.Optional[tp.Any] = None,
    ):

        if self.initialized:
            return

        method = self.call_init_step if self.run_eagerly else self.call_init_step_jit
        states = self.states.copy() if self.run_eagerly else self.states

        state_updates = method(
            x,
            y_true,
            sample_weight,
            class_weight,
            states,
        )

        self.states = self.states.maybe_update(**state_updates)
        self.initial_states = self.initial_states.maybe_update(**state_updates)
        self.initialized = True

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
        if not self.initialized:
            raise types.ModelNotInitialized(
                f"Model not initialized, please execute `init` or `init_on_batch` before running this method."
            )
        initializing = False
        training = False

        method = self.call_pred_step if self.run_eagerly else self.call_pred_step_jit
        states = self.states.copy() if self.run_eagerly else self.states

        y_pred, self.states = method(
            x,
            states,
            initializing,
            training,
        )

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
        self.init_on_batch(
            x=x,
            y_true=y,
            sample_weight=sample_weight,
            class_weight=class_weight,
        )

        initializing = False
        training = False

        method = self.call_test_step if self.run_eagerly else self.call_test_step_jit
        states = self.states.copy() if self.run_eagerly else self.states

        loss, logs, self.states = method(
            x,
            y,
            sample_weight,
            class_weight,
            states,
            initializing,
            training,
        )

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
        self.init_on_batch(
            x=x,
            y_true=y,
            sample_weight=sample_weight,
            class_weight=class_weight,
        )

        initializing = False
        training = True

        method = self.call_train_step if self.run_eagerly else self.call_train_step_jit
        states = self.states.copy() if self.run_eagerly else self.states

        logs, self.states = method(
            x,
            y,
            sample_weight,
            class_weight,
            states,
            initializing,
            training,
        )

        return logs

    def summary(
        self,
        x: tp.Optional[tp.Any] = None,
        depth: int = 2,
        return_repr: bool = False,
        initialize: bool = False,
        eval_shape: bool = True,
    ) -> tp.Optional[str]:
        """
        Prints a summary of the network.
        Arguments:
            x: A sample of inputs to the network.
            depth: The level number of nested level which will be showed.
                Information about summaries from modules deeper than `depth`
                will be aggregated together.
        """
        raise NotImplementedError()

    def save(
        self,
        path: tp.Union[str, pathlib.Path],
    ) -> None:
        """
        Saves the model to disk.

        It creates a directory that includes:

        - `{path}/model.pkl`: The `Model` object instance serialized with `cloudpickle`,
            this allows you to re-instantiate the model later.
        - `{path}/states.pkl`: The `Model.states` serialized with `cloudpickle`.
        - `{path}/initial_states.pkl`: The `Model.initial_states` serialized with `cloudpickle`.

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

        (path / "states.pkl").write_bytes(cloudpickle.dumps(self.states))

        (path / "initial_states.pkl").write_bytes(
            cloudpickle.dumps(self.initial_states)
        )

        (path / "model.pkl").write_bytes(cloudpickle.dumps(self))

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
        if isinstance(path, str):
            path = pathlib.Path(path)

        self.states = cloudpickle.loads((path / "states.pkl").read_bytes())
        self.initial_states = cloudpickle.loads(
            (path / "initial_states.pkl").read_bytes()
        )

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

        states = types.States(
            {field: value for field, value in self.states.items() if value is not None}
        )
        flat_states, states_def = jax.tree_flatten(states)

        def jax_fn(flat_states, inputs):
            states = jax.tree_unflatten(states_def, flat_states)

            y_pred, _ = utils.inject_dependencies(self.pred_step)(
                x=inputs, states=states, initializing=False, training=False
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

    def reset(self):
        self.states = self.initial_states.copy()

    def reset_metrics(self):
        if hasattr(self.initial_states, "metrics_states"):
            self.states = self.states.update(
                metrics_states=self.initial_states.metrics_states
            )
