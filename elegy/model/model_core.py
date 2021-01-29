from copy import copy
import pickle
import threading
import typing as tp
from dataclasses import dataclass
from enum import Enum
from io import StringIO
import pathlib
from contextlib import contextmanager

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
    Grads,
    Scalar,
    Summaries,
    Logs,
    RNG,
    RNGSeq,
    States,
    UNINITIALIZED,
    Uninitialized,
    Protocol,
)
from elegy.types import Mode
from tabulate import tabulate


class LocalContext(Protocol):
    mode: tp.Optional[Mode]


@dataclass
class _LocalContext(threading.local):
    mode: tp.Optional[Mode]


LOCAL: LocalContext = _LocalContext(mode=None)


class PredStep(tp.NamedTuple):
    y_pred: tp.Any
    states: States
    aux_losses: Logs
    aux_metrics: Logs
    summaries: Summaries

    @classmethod
    def simple(cls, y_pred: tp.Any, states: States):
        return cls(
            y_pred=y_pred,
            states=states,
            aux_losses=hooks.get_losses(),
            aux_metrics=hooks.get_metrics(),
            summaries=hooks.get_summaries(),
        )


class TestStep(tp.NamedTuple):
    loss: Scalar
    logs: Logs
    states: States


class GradStep(tp.NamedTuple):
    loss: Scalar
    logs: Logs
    states: States
    grads: Grads


class TrainStep(tp.NamedTuple):
    logs: Logs
    states: States


class ModelCore:
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
        self.call_pred_step_jit = hooks.jit(
            self.call_pred_step,
            static_argnums=[2, 3],
        )
        self.call_test_step_jit = hooks.jit(
            self.call_test_step,
            static_argnums=[5, 6],
        )
        self.call_train_step_jit = hooks.jit(
            self.call_train_step,
            static_argnums=[5],
        )

    def __setstate__(self, d):
        self.__dict__ = d
        self._jit_functions()

    def __getstate__(self):
        d = self.__dict__.copy()

        # remove states
        del d["states"]
        del d["initial_states"]

        # remove jitted functions
        del d["call_pred_step_jit"]
        del d["call_test_step_jit"]
        del d["call_train_step_jit"]

        return d

    # ----------------------------------------------------------------
    # Abstract API
    # ----------------------------------------------------------------

    def pred_step(
        self,
        net_params: tp.Any,
        x: tp.Any,
        net_states: tp.Any,
        rng: tp.Any,
        states: States,
        training: bool,
        initializing: bool,
    ) -> PredStep:
        raise NotImplementedError()

    def call_pred_step(
        self,
        x: tp.Any,
        states: States,
        training: bool,
        initializing: bool,
    ) -> PredStep:
        assert LOCAL.mode is not None

        losses = metrics = LOCAL.mode in (Mode.test, Mode.train)
        summaries = LOCAL.mode == Mode.summary

        with hooks.context(losses=losses, metrics=metrics, summaries=summaries):
            return utils.inject_dependencies(self.pred_step)(
                net_params=states.net_params,
                x=x,
                net_states=states.net_states,
                rng=states.rng,
                states=states,
                training=training,
                initializing=initializing,
            )

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
        states: States,
        training: bool,
        initializing: bool,
    ) -> TestStep:
        raise NotImplementedError()

    def call_test_step(
        self,
        x: tp.Any,
        y_true: tp.Any,
        sample_weight: tp.Optional[np.ndarray],
        class_weight: tp.Optional[np.ndarray],
        states: States,
        training: bool,
        initializing: bool,
    ) -> TestStep:
        return utils.inject_dependencies(self.test_step)(
            net_params=states.net_params,
            x=x,
            y_true=y_true,
            net_states=states.net_states,
            metrics_states=states.metrics_states,
            sample_weight=sample_weight,
            class_weight=class_weight,
            rng=states.rng,
            states=states,
            training=training,
            initializing=initializing,
        )

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
        states: States,
        training: bool,
        initializing: bool,
    ) -> GradStep:
        raise NotImplementedError()

    def call_grad_step(
        self,
        x: tp.Any,
        y_true: tp.Any,
        sample_weight: tp.Optional[np.ndarray],
        class_weight: tp.Optional[np.ndarray],
        states: States,
        training: bool,
        initializing: bool,
    ) -> GradStep:
        return utils.inject_dependencies(self.grad_step)(
            net_params=states.net_params,
            x=x,
            y_true=y_true,
            net_states=states.net_states,
            metrics_states=states.metrics_states,
            sample_weight=sample_weight,
            class_weight=class_weight,
            rng=states.rng,
            states=states,
            training=training,
            initializing=initializing,
        )

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
        states: States,
        training: bool,
        initializing: bool,
    ) -> TrainStep:
        raise NotImplementedError()

    def call_train_step(
        self,
        x: tp.Any,
        y_true: tp.Any,
        sample_weight: tp.Optional[np.ndarray],
        class_weight: tp.Optional[np.ndarray],
        states: States,
        initializing: bool,
    ) -> TrainStep:
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
            states=states,
            training=True,
            initializing=initializing,
        )

    # ----------------------------------------------------------------
    # *_on_batch methods
    # ----------------------------------------------------------------

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
        with model_context(Mode.pred):
            self.maybe_initialize(x=x)

            method = (
                self.call_pred_step if self.run_eagerly else self.call_pred_step_jit
            )

            training = False
            initializing = False

            y_pred, state_updates, _, _, _ = method(
                x, self.states, training, initializing
            )

        self.states = self.states.merge(state_updates)

        return y_pred

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
        with model_context(Mode.test):
            self.maybe_initialize(
                x=x,
                y_true=y,
                sample_weight=sample_weight,
                class_weight=class_weight,
            )

            method = (
                self.call_test_step if self.run_eagerly else self.call_test_step_jit
            )

            training = False
            initializing = False

            loss, logs, state_updates = method(
                x,
                y,
                sample_weight,
                class_weight,
                self.states,
                training,
                initializing,
            )

        self.states = self.states.merge(state_updates)

        return logs

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

        with model_context(Mode.train):
            self.maybe_initialize(
                x=x,
                y_true=y,
                sample_weight=sample_weight,
                class_weight=class_weight,
            )

            method = (
                self.call_train_step if self.run_eagerly else self.call_train_step_jit
            )

            initializing = False

            logs, state_updates = method(
                x,
                y,
                sample_weight,
                class_weight,
                self.states,
                initializing,
            )

        self.states = self.states.merge(state_updates)

        return logs

    # ----------------------------------------------------------------
    # other methods
    # ----------------------------------------------------------------

    def update_modules(self):
        pass

    def reset(self):
        self.states = self.initial_states.copy()

    def reset_metrics(self):
        self.states = self.states.update(
            metrics_states=self.initial_states.metrics_states
        )

    def maybe_initialize(
        self,
        x: tp.Union[np.ndarray, tp.Mapping[str, tp.Any], tp.Tuple] = (),
        y_true: tp.Union[np.ndarray, tp.Mapping[str, tp.Any], tp.Tuple, None] = None,
        sample_weight: tp.Optional[np.ndarray] = None,
        class_weight: tp.Optional[np.ndarray] = None,
    ):
        assert LOCAL.mode is not None

        mode = LOCAL.mode
        rng = self.states.rng if isinstance(self.states.rng, RNGSeq) else None
        training = True
        initializing = True
        state_updates: States

        if (
            mode in (Mode.pred, Mode.summary)
            and isinstance(self.states.net_params, Uninitialized)
            and isinstance(self.states.net_states, Uninitialized)
        ):
            method = (
                self.call_pred_step if self.run_eagerly else self.call_pred_step_jit
            )

            _, state_updates, _, _, _ = method(
                x,
                self.states,
                training,
                initializing,
            )
        elif mode == Mode.test and isinstance(
            self.states.metrics_states, Uninitialized
        ):
            method = (
                self.call_test_step if self.run_eagerly else self.call_test_step_jit
            )

            _, _, state_updates = method(
                x,
                y_true,
                sample_weight,
                class_weight,
                self.states,
                training,
                initializing,
            )
        elif mode == Mode.train and isinstance(
            self.states.optimizer_states, Uninitialized
        ):
            method = (
                self.call_train_step if self.run_eagerly else self.call_train_step_jit
            )

            _, state_updates = method(
                x,
                y_true,
                sample_weight,
                class_weight,
                self.states,
                initializing,
            )

        else:
            return

        if mode in (Mode.pred, Mode.test, Mode.train):
            if isinstance(state_updates.net_params, Uninitialized):
                state_updates = state_updates.update(net_params=None)

            if isinstance(state_updates.net_states, Uninitialized):
                state_updates = state_updates.update(net_states=None)
        if mode in (Mode.test, Mode.train):
            if isinstance(state_updates.metrics_states, Uninitialized):
                state_updates = state_updates.update(metrics_states=None)

        if mode == Mode.train:
            if isinstance(state_updates.optimizer_states, Uninitialized):
                state_updates = state_updates.update(optimizer_states=None)

        self.states = self.states.merge_new(state_updates)
        self.initial_states = self.initial_states.merge_new(state_updates)

        # update modules
        self.update_modules()

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
            (path / "initial_states.pkl").read_bytes()
        )


# ----------------------------------------------------------------
# context managers
# ---------------------------------------------------------------


def model_context(mode: Mode) -> tp.ContextManager[None]:
    return _model_context(mode)


@contextmanager
def _model_context(mode: Mode):
    prev_mode = LOCAL.mode

    LOCAL.mode = mode

    try:
        yield
    finally:
        LOCAL.mode = prev_mode
