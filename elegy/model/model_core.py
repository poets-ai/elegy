from copy import copy
import pickle
import typing as tp
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from io import StringIO
import pathlib

import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from elegy import hooks, module, utils
from elegy.losses.loss import Loss
from elegy.metrics.metric import Metric
from elegy.types import (
    Backprop,
    Evaluation,
    Logs,
    Prediction,
    RNG,
    RNGSeq,
    States,
    Training,
    UNINITIALIZED,
    Uninitialized,
)
from elegy.types import Mode
from tabulate import tabulate


class ModelCore(ABC):
    states: States
    initial_states: States
    history: tp.Dict[str, tp.Any]
    run_eagerly: bool = False

    def __init__(
        self,
        net_params: tp.Union[Uninitialized, tp.Any] = UNINITIALIZED,
        net_states: tp.Union[Uninitialized, tp.Any] = UNINITIALIZED,
        metrics_states: tp.Union[Uninitialized, tp.Any] = UNINITIALIZED,
        optimizer_states: tp.Union[Uninitialized, tp.Any] = UNINITIALIZED,
        rng: tp.Union[Uninitialized, tp.Any] = UNINITIALIZED,
        run_eagerly: bool = False,
    ):
        self.states = States(
            net_params=net_params,
            net_states=net_states,
            metrics_states=metrics_states,
            optimizer_states=optimizer_states,
            rng=rng,
        )
        self.initial_states = self.states.copy()
        self.run_eagerly = run_eagerly
        self.history = {}

        self._jit_functions()

    def _jit_functions(self):
        self.init_internal_jit = hooks.jit(
            self.init_internal,
            static_argnums=[0],
        )
        self.pred_step_internal_jit = hooks.jit(
            self.pred_step_internal,
            static_argnums=[2],
        )
        self.test_step_internal_jit = hooks.jit(
            self.test_step_internal,
            static_argnums=[5],
        )
        self.train_step_internal_jit = hooks.jit(
            self.train_step_internal,
        )

    def __setstate__(self, d):
        self.__dict__ = d
        self._jit_functions()

    def __getstate__(self):
        d = self.__dict__.copy()

        # remove states
        # del d["states"]
        # del d["initial_states"]

        # remove jitted functions
        del d["init_internal_jit"]
        del d["pred_step_internal_jit"]
        del d["test_step_internal_jit"]
        del d["train_step_internal_jit"]

        return d

    @abstractmethod
    def init(
        self,
        mode: Mode,
        x: tp.Any,
        y_true: tp.Any,
        sample_weight: tp.Optional[np.ndarray],
        class_weight: tp.Optional[np.ndarray],
    ) -> States:
        ...

    @abstractmethod
    def pred_step(
        self,
        net_params: tp.Any,
        x: tp.Any,
        net_states: tp.Any,
        rng: tp.Any,
        training: bool,
        states: States,
    ) -> Prediction:
        ...

    @abstractmethod
    def test_step(
        self,
        net_params: tp.Any,
        x: tp.Any,
        y_true: tp.Any,
        net_states: tp.Any,
        metrics_states: tp.Any,
        sample_weight: tp.Optional[np.ndarray],
        class_weight: tp.Optional[np.ndarray],
        rng: tp.Any,
        training: bool,
        states: States,
    ) -> Evaluation:
        ...

    def grad_step(
        self,
        net_params: tp.Any,
        x: tp.Any,
        y_true: tp.Any,
        net_states: tp.Any,
        metrics_states: tp.Any,
        sample_weight: tp.Optional[np.ndarray],
        class_weight: tp.Optional[np.ndarray],
        rng: tp.Any,
        training: bool,
        states: States,
    ) -> Backprop:
        raise NotImplementedError()

    @abstractmethod
    def train_step(
        self,
        net_params: tp.Any,
        x: tp.Any,
        y_true: tp.Any,
        net_states: tp.Any,
        metrics_states: tp.Any,
        optimizer_states: tp.Any,
        sample_weight: tp.Optional[np.ndarray],
        class_weight: tp.Optional[np.ndarray],
        rng: tp.Any,
        training: bool,
        states: States,
    ) -> Training:
        ...

    def reset(self):
        self.states = self.initial_states.copy()

    def reset_metrics(self):
        self.states = self.states.update(
            metrics_states=self.initial_states.metrics_states
        )

    def init_internal(
        self,
        mode: Mode,
        x: tp.Any,
        y_true: tp.Any,
        sample_weight: tp.Optional[np.ndarray],
        class_weight: tp.Optional[np.ndarray],
    ) -> States:
        return utils.inject_dependencies(self.init)(
            mode=mode,
            x=x,
            y_true=y_true,
            sample_weight=sample_weight,
            class_weight=class_weight,
        )

    def maybe_initialize(
        self,
        mode: Mode,
        x: tp.Union[np.ndarray, tp.Mapping[str, tp.Any], tp.Tuple] = (),
        y_true: tp.Union[np.ndarray, tp.Mapping[str, tp.Any], tp.Tuple, None] = None,
        sample_weight: tp.Optional[np.ndarray] = None,
        class_weight: tp.Optional[np.ndarray] = None,
    ):

        if (
            (
                mode == Mode.pred
                and isinstance(self.states.net_params, Uninitialized)
                and isinstance(self.states.net_states, Uninitialized)
            )
            or (
                mode == Mode.test
                and isinstance(self.states.metrics_states, Uninitialized)
            )
            or (
                mode == Mode.train
                and isinstance(self.states.optimizer_states, Uninitialized)
            )
        ):
            method = self.init_internal if self.run_eagerly else self.init_internal_jit

            state_updates: States = method(
                mode,
                x,
                y_true,
                sample_weight,
                class_weight,
            )

            self.states = self.states.merge_new(state_updates)
            self.initial_states = self.initial_states.merge_new(state_updates)

    def pred_step_internal(
        self,
        states: States,
        x: tp.Any,
        training: bool,
    ) -> tp.Tuple[tp.Any, States]:
        return utils.inject_dependencies(self.pred_step)(
            net_params=states.net_params,
            x=x,
            net_states=states.net_states,
            rng=states.rng,
            training=training,
            states=states,
        )

    def predict_on_batch(
        self, x: tp.Union[np.ndarray, tp.Mapping[str, tp.Any], tp.Tuple]
    ) -> tp.Union[np.ndarray, tp.Mapping[str, tp.Any], tp.Tuple]:
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
        self.maybe_initialize(mode=Mode.pred, x=x)

        method = (
            self.pred_step_internal if self.run_eagerly else self.pred_step_internal_jit
        )

        training = False
        rng = self.states.rng if isinstance(self.states.rng, RNGSeq) else None

        with hooks.context(rng=rng, initializing=False, training=False):
            y_pred, state_updates = method(
                self.states,
                x,
                training,
            )

        self.states = self.states.merge(state_updates)

        return y_pred

    def test_step_internal(
        self,
        states: States,
        x: tp.Any,
        y_true: tp.Any,
        sample_weight: tp.Optional[np.ndarray],
        class_weight: tp.Optional[np.ndarray],
        training: bool,
    ) -> Evaluation:
        return utils.inject_dependencies(self.test_step)(
            net_params=states.net_params,
            x=x,
            y_true=y_true,
            net_states=states.net_states,
            metrics_states=states.metrics_states,
            sample_weight=sample_weight,
            class_weight=class_weight,
            rng=states.rng,
            training=training,
            states=states,
        )

    def test_on_batch(
        self,
        x: tp.Union[np.ndarray, tp.Mapping[str, tp.Any], tp.Tuple],
        y: tp.Union[np.ndarray, tp.Mapping[str, tp.Any], tp.Tuple, None] = None,
        sample_weight: tp.Optional[np.ndarray] = None,
        class_weight: tp.Optional[np.ndarray] = None,
    ) -> Logs:
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
        self.maybe_initialize(
            mode=Mode.test,
            x=x,
            y_true=y,
            sample_weight=sample_weight,
            class_weight=class_weight,
        )

        method = (
            self.test_step_internal if self.run_eagerly else self.test_step_internal_jit
        )

        training = False
        rng = self.states.rng if isinstance(self.states.rng, RNGSeq) else None

        with hooks.context(
            rng=rng, losses=True, metrics=True, initializing=False, training=False
        ):
            loss, logs, state_updates = method(
                self.states,
                x,
                y,
                sample_weight,
                class_weight,
                training,
            )

        self.states = self.states.merge(state_updates)

        return logs

    def grad_step_internal(
        self,
        states: States,
        x: tp.Any,
        y_true: tp.Any,
        sample_weight: tp.Optional[np.ndarray],
        class_weight: tp.Optional[np.ndarray],
        training: bool,
    ) -> Backprop:
        return utils.inject_dependencies(self.grad_step)(
            net_params=states.net_params,
            x=x,
            y_true=y_true,
            net_states=states.net_states,
            metrics_states=states.metrics_states,
            sample_weight=sample_weight,
            class_weight=class_weight,
            rng=states.rng,
            training=training,
            states=states,
        )

    def train_step_internal(
        self,
        states: States,
        x: tp.Any,
        y_true: tp.Any,
        sample_weight: tp.Optional[np.ndarray],
        class_weight: tp.Optional[np.ndarray],
    ) -> Training:
        return utils.inject_dependencies(self.train_step)(
            net_params=states.net_params,
            x=x,
            y_true=y_true,
            net_states=states.net_states,
            metrics_states=states.metrics_states,
            optimizer_states=states.optimizer_states,
            sample_weight=sample_weight,
            class_weight=class_weight,
            rng=states.rng,
            training=True,
            states=states,
        )

    def train_on_batch(
        self,
        x: tp.Union[np.ndarray, tp.Mapping[str, tp.Any], tp.Tuple],
        y: tp.Union[np.ndarray, tp.Mapping[str, tp.Any], tp.Tuple, None] = None,
        sample_weight: tp.Optional[np.ndarray] = None,
        class_weight: tp.Optional[np.ndarray] = None,
    ) -> Logs:
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
        self.maybe_initialize(
            mode=Mode.train,
            x=x,
            y_true=y,
            sample_weight=sample_weight,
            class_weight=class_weight,
        )

        method = (
            self.train_step_internal
            if self.run_eagerly
            else self.train_step_internal_jit
        )

        rng = self.states.rng if isinstance(self.states.rng, RNGSeq) else None

        with hooks.context(
            rng=rng, losses=True, metrics=True, initializing=False, training=True
        ):
            logs, state_updates = method(
                self.states,
                x,
                y,
                sample_weight,
                class_weight,
            )

        self.states = self.states.merge(state_updates)

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
        - `{path}/states.pkl`: The `Model.states` serialized with `pickle`.
        - `{path}/initial_states.pkl`: The `Model.initial_states` serialized with `pickle`.

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

        with open(path / "states.pkl", "wb") as f:
            cloudpickle.dump(self.states, f)

        with open(path / "initial_states.pkl", "wb") as f:
            cloudpickle.dump(self.initial_states, f)

        path = path / "model.pkl"

        with open(path, "wb") as f:
            try:
                cloudpickle.dump(self, f)
            except BaseException as e:
                print(
                    f"Error occurred saving the model object at {path}\nContinuing...."
                )

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
            (path / "intial_states.pkl").read_bytes()
        )
