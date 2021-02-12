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
    init_stage: types.Mode = types.Mode.none

    def __init__(
        self,
        states: tp.Optional[types.States] = None,
        run_eagerly: bool = False,
        init_stage: types.Mode = types.Mode.none,
    ):

        base_states = self.states_step()

        if states is None:
            states = types.States()

        states = states.maybe_update(**base_states)

        self.initial_states = states
        self.states = states.copy()  # explicity do this to copy RNGSeq
        self.run_eagerly = run_eagerly
        self.history = {}
        self.init_stage = init_stage
        self.jitted_members: tp.Set[str] = set()

        self.jit_step()

    def jit_step(self):
        self.call_summary_step_jit = jax.jit(
            self.call_summary_step,
            static_argnums=[2, 3],
        )
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
            static_argnums=[0],
        )

        self.jitted_members |= {
            "call_summary_step_jit",
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
        mode: types.Mode,
        x: tp.Any,
        y_true: tp.Any,
        sample_weight: tp.Optional[np.ndarray],
        class_weight: tp.Optional[np.ndarray],
        states: types.States,
    ) -> tp.Union[tp.Tuple[types.Mode, types.States], types.States]:
        training = True
        initializing = True

        if mode == types.Mode.pred:
            _, states = utils.inject_dependencies(self.pred_step)(
                x=x,
                states=states,
                initializing=initializing,
                training=training,
            )
        elif mode == types.Mode.test:
            _, _, states = utils.inject_dependencies(self.test_step)(
                x=x,
                y_true=y_true,
                sample_weight=sample_weight,
                class_weight=class_weight,
                states=states,
                initializing=initializing,
                training=training,
            )
        elif mode == types.Mode.train:
            _, states = utils.inject_dependencies(self.train_step)(
                x=x,
                y_true=y_true,
                sample_weight=sample_weight,
                class_weight=class_weight,
                states=states,
                initializing=initializing,
                training=training,
            )
        else:
            raise ValueError(f"Invalid mode '{mode}'")

        return mode, states

    def call_init_step(
        self,
        mode: types.Mode,
        x: tp.Any,
        y_true: tp.Any,
        sample_weight: tp.Optional[np.ndarray],
        class_weight: tp.Optional[np.ndarray],
        states: types.States,
    ) -> tp.Union[tp.Tuple[types.Mode, types.States], types.States]:
        return utils.inject_dependencies(self.init_step)(
            mode=mode,
            x=x,
            y_true=y_true,
            sample_weight=sample_weight,
            class_weight=class_weight,
            states=states,
        )

    def summary_step(
        self,
        x: tp.Any,
        states: types.States,
        initializing: bool,
        training: bool,
    ) -> tp.List[types.SummaryTableEntry]:
        raise NotImplementedError()

    def call_summary_step(
        self,
        x: tp.Any,
        states: types.States,
        initializing: bool,
        training: bool,
    ) -> tp.List[types.SummaryTableEntry]:
        return utils.inject_dependencies(self.summary_step)(
            x=x,
            states=states,
            initializing=initializing,
            training=training,
        )

    def pred_step(
        self,
        x: tp.Any,
        states: types.States,
        initializing: bool,
        training: bool,
    ) -> PredStep:
        raise NotImplementedError()

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
        raise NotImplementedError()

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
        raise NotImplementedError()

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
        raise NotImplementedError()

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
        mode = types.Mode.train
        self.maybe_initialize(mode, x, y_true, sample_weight, class_weight)

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
        mode = types.Mode.pred
        initializing = False
        training = False

        self.maybe_initialize(mode=mode, x=x)

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
        mode = types.Mode.test
        initializing = False
        training = False

        self.maybe_initialize(
            mode,
            x=x,
            y_true=y,
            sample_weight=sample_weight,
            class_weight=class_weight,
        )

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
        mode = types.Mode.train
        initializing = False
        training = True

        self.maybe_initialize(
            mode=mode,
            x=x,
            y_true=y,
            sample_weight=sample_weight,
            class_weight=class_weight,
        )

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
        x,
        depth: int = 2,
        tablefmt: str = "fancy_grid",
        return_repr: bool = False,
        **tablulate_kwargs,
    ) -> tp.Optional[str]:
        """
        Prints a summary of the network.
        Arguments:
            x: A sample of inputs to the network.
            depth: The level number of nested level which will be showed.
                Information about summaries from modules deeper than `depth`
                will be aggregated together.
            tablefmt: A string representing the style of the table generated by
                `tabulate`. See
                [python-tabulate](https://github.com/astanin/python-tabulate)
                for more options.
            tablulate_kwargs: Additional keyword arguments passed to `tabulate`.
                See [python-tabulate](https://github.com/astanin/python-tabulate)
                for more options.
        """
        mode = types.Mode.pred
        initializing = False
        training = False

        self.maybe_initialize(mode, x=x)

        method = (
            self.call_summary_step if self.run_eagerly else self.call_summary_step_jit
        )
        states = self.states.copy() if self.run_eagerly else self.states

        entries = method(
            x,
            states,
            initializing,
            training,
        )
        total_entry = entries[-1]
        entries = entries[:-1]

        depth_groups: tp.Dict[str, tp.List[types.SummaryTableEntry]] = toolz.groupby(
            lambda entry: "/".join(entry.path.split("/")[:depth]), entries
        )

        def get_grouped_entry(
            entry: types.SummaryTableEntry,
        ) -> types.SummaryTableEntry:
            group = depth_groups[entry.path]

            return types.SummaryTableEntry(
                path=entry.path,
                module_type_name=entry.module_type_name,
                output_value=entry.output_value,
                trainable_params_count=sum(
                    entry_.trainable_params_count for entry_ in group
                ),
                trainable_params_size=sum(
                    entry_.trainable_params_size for entry_ in group
                ),
                non_trainable_params_count=sum(
                    entry_.non_trainable_params_count for entry_ in group
                ),
                non_trainable_params_size=sum(
                    entry_.non_trainable_params_size for entry_ in group
                ),
            )

        entries = [
            get_grouped_entry(entry) for entry in entries if entry.path in depth_groups
        ]

        def format_output(value) -> str:
            file = StringIO()
            outputs = jax.tree_map(lambda x: f"{x.shape}{{pad}}  {x.dtype}", value)
            yaml.safe_dump(
                outputs, file, default_flow_style=False, indent=2, explicit_end=False
            )
            return file.getvalue().replace("\n...", "")

        def format_size(size):
            return (
                f"{size / 1e9 :,.1f} GB"
                if size > 1e9
                else f"{size / 1e6 :,.1f} MB"
                if size > 1e6
                else f"{size / 1e3 :,.1f} KB"
                if size > 1e3
                else f"{size:,} B"
            )

        table: tp.List = [
            [
                "Inputs",
                format_output(x),
                "",
                "",
            ]
        ]

        for entry in entries:

            table.append(
                [
                    f"{entry.path}{{pad}}  {entry.module_type_name}",
                    format_output(entry.output_value),
                    f"{entry.trainable_params_count:,}{{pad}}    {format_size(entry.trainable_params_size)}"
                    if entry.trainable_params_count > 0
                    else "",
                    f"{entry.non_trainable_params_count:,}{{pad}}    {format_size(entry.non_trainable_params_size)}"
                    if entry.non_trainable_params_count > 0
                    else "",
                ]
            )

        # add padding
        for col in range(4):
            max_length = max(
                len(line.split("{pad}")[0])
                for row in table
                for line in row[col].split("\n")
            )

            for row in table:
                row[col] = "\n".join(
                    line.format(
                        pad=" " * (max_length - len(line.rstrip().split("{pad}")[0]))
                    )
                    for line in row[col].rstrip().split("\n")
                )

        # global summaries
        params_count = total_entry.trainable_params_count
        params_size = total_entry.trainable_params_size
        states_count = total_entry.non_trainable_params_count
        states_size = total_entry.non_trainable_params_size
        total_count = params_count + states_count
        total_size = params_size + states_size

        summary = (
            "\n"
            + tabulate(
                table,
                headers=[
                    "Layer",
                    "Outputs Shape",
                    "Trainable\nParameters",
                    "Non-trainable\nParameters",
                ],
                tablefmt=tablefmt,
                **tablulate_kwargs,
            )
            + "\n"
            + tabulate(
                [
                    [
                        f"Total Parameters:",
                        f"{total_count:,}",
                        f"{format_size(total_size)}" if total_count > 0 else "",
                    ],
                    [
                        f"Trainable Parameters:",
                        f"{params_count:,}",
                        f"{format_size(params_size)}" if params_count > 0 else "",
                    ],
                    [
                        f"Non-trainable Parameters:",
                        f"{states_count:,}",
                        f"{format_size(states_size)}" if states_count > 0 else "",
                    ],
                ],
                tablefmt="plain",
            )
            + "\n"
        )

        print(summary)

        if return_repr:
            return summary

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
        if isinstance(path, str):
            path = pathlib.Path(path)

        self.states = cloudpickle.loads((path / "states.pkl").read_bytes())
        self.initial_states = cloudpickle.loads(
            (path / "initial_states.pkl").read_bytes()
        )

    def reset(self):
        self.states = self.initial_states.copy()

    def reset_metrics(self):
        if hasattr(self.initial_states, "metrics_states"):
            self.states = self.states.update(
                metrics_states=self.initial_states.metrics_states
            )

    # ----------------------------------------------------------------
    # other methods
    # ----------------------------------------------------------------

    def maybe_initialize(
        self,
        mode: types.Mode,
        x: tp.Any = (),
        y_true: tp.Any = None,
        sample_weight: tp.Optional[np.ndarray] = None,
        class_weight: tp.Optional[tp.Any] = None,
    ):

        if mode <= self.init_stage:
            return

        state_updates: types.States

        method = self.call_init_step if self.run_eagerly else self.call_init_step_jit
        states = self.states.copy() if self.run_eagerly else self.states

        output = method(
            mode,
            x,
            y_true,
            sample_weight,
            class_weight,
            states,
        )

        if isinstance(output, tuple):
            mode, state_updates = output
        else:
            mode = types.Mode.train
            state_updates = output

        if mode > self.init_stage:
            self.init_stage = mode

        self.states = self.states.maybe_update(**state_updates)
        self.initial_states = self.initial_states.maybe_update(**state_updates)
