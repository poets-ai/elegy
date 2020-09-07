# Implementation based on tf.keras.engine.training.py
# https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/engine/training.py

import re
from elegy import types
import functools
from elegy.module import (
    ApplyCallable,
    ApplyContext,
    Module,
    PRNGSequence,
)
from io import StringIO
import json
import logging
import pickle
import typing as tp
from copy import copy
from enum import Enum
from functools import partial
from pathlib import Path

import cloudpickle
import deepdish
import jax
import jax.numpy as jnp
import numpy as np
import toolz
import optax
from tabulate import tabulate
import yaml

from elegy.losses import loss_modes
from elegy.metrics import metric_modes

from . import utils
from .callbacks import Callback, CallbackList, History
from .data import (
    DataHandler,
    map_append,
    map_structure,
    train_validation_split,
    unpack_x_y_sample_weight,
)


__all__ = ["Model", "load"]


class Mode(Enum):
    predict = 1
    test = 2
    train = 3


class Model:
    """
    `Model` is tasked with performing training, evaluation, and inference for a given
    `elegy.Module` or `haiku.Module`.

    To create a `Model` you first have to define its architecture in a `Module`:
    ```python
    class MLP(elegy.Module):
        def call(self, image: jnp.ndarray) -> jnp.ndarray:
            mlp = hk.Sequential([
                hk.Flatten(),
                hk.Linear(300),
                jax.nn.relu,
                hk.Linear(10),
            ])
            return mlp(image)
    ```

    Then you can pass this `Module` to the `Model`'s constructor and specify additional things like losses, metrics, optimizer, and callbacks:
    ```python
    model = elegy.Model(
        module=MLP(),
        loss=[
            elegy.losses.SparseCategoricalCrossentropy(from_logits=True),
            elegy.regularizers.GlobalL2(l=1e-5),
        ],
        metrics=elegy.metrics.SparseCategoricalAccuracy(),
        optimizer=optax.rmsprop(1e-3),
    )
    ```

    Once the model is created, you can train the model with `model.fit()`, or use the model
    to do prediction with `model.predict()`.
    Checkout [Getting Started](https://poets-ai.github.io/elegy/getting-started) for
    additional details.

    Attributes:
        parameters: A `haiku.Params` structure with the weights of the model.
        states: A `haiku.State` structure with non-trainable parameters of the model.
        optimizer_state:  A `optax.OptState` structure with states of the optimizer.
        metrics_states: A `haiku.State` structure with the states of the metrics.
        initial_metrics_state: A `haiku.State` structure with the initial states of the metrics.
        run_eagerly: Settable attribute indicating whether the model should run eagerly.
            Running eagerly means that your model will be run step by step, like Python code, instead of
            using Jax's `jit` to optimize the computation. Your model might run slower, but it should become easier for you to debug
            it by stepping into individual layer calls.
    """

    # public fields
    module: Module
    optimizer_state: tp.Optional[optax.OptState]
    initial_metrics_state: tp.Optional[tp.Dict]
    run_eagerly: bool

    # private fields
    loss_module: tp.Optional[Module]
    metrics_module: tp.Optional[Module]
    _optimizer: optax.GradientTransformation
    _rngs: PRNGSequence
    _parameters: tp.Optional[types.Parameters]
    _states: tp.Optional[types.States]
    _metrics_states: tp.Optional[tp.Dict]

    __all__ = [
        "evaluate",
        "fit",
        "load",
        "predict",
        "predict_on_batch",
        "reset",
        "reset_metrics",
        "save",
        "summary",
        "test_on_batch",
        "train_on_batch",
        "full_state",
        "parameters",
        "seed",
        "states",
    ]

    def __init__(
        self,
        module: Module,
        loss: tp.Union[tp.Callable, tp.List, tp.Dict, None] = None,
        metrics: tp.Union[tp.Callable, tp.List, tp.Dict, None] = None,
        optimizer: tp.Optional[optax.GradientTransformation] = None,
        run_eagerly: bool = False,
        parameters: tp.Optional[types.Parameters] = None,
        states: tp.Optional[types.States] = None,
        optimizer_state: tp.Optional[optax.OptState] = None,
        metrics_states: tp.Optional[tp.Dict] = None,
        initial_metrics_state: tp.Optional[tp.Dict] = None,
        seed: tp.Union[np.ndarray, int] = 42,
    ):
        """[summary]

        Arguments:
            module: A `Module` instance.
            loss: A `elegy.Loss` or `Callable` instance representing the loss function of the network.
                You can define more loss terms by simply passing a possibly nested structure of
                lists and dictionaries of `elegy.Loss` or `Callable`s. Usually a plain list of losses is enough
                but using dictionaries will create namescopes for the names of the losses
                which might be useful e.g. to group things in tensorboard. Contrary to Keras convention,
                in Elegy there is no relation between the structure of `loss` with the structure
                of the labels and outputs of the network. Elegy's loss system is more flexible than
                the one provided by Keras, for more information on how to mimick Keras behavior checkout the
                [Losses and Metrics Guide](https://poets-ai.github.io/elegy/guides/losses-and-metrics)`.
            metrics: A `elegy.Metric` or `Callable` instance representing the loss function of the network.
                You can define more metrics terms by simply passing a possibly nested structure of
                lists and dictionaries of `elegy.Metric` or `Callable`s. Usually a plain list of metrics is enough
                but using dictionaries will create namescopes for the names of the metrics
                which might be useful e.g. to group things in tensorboard. Contrary to Keras convention,
                in Elegy there is no relation between the structure of `metrics` with the structure
                of the labels and outputs of the network. Elegy's metrics system is more flexible than
                the one provided by Keras, for more information on how to mimick Keras behavior checkout the
                [Losses and Metrics Guide](https://poets-ai.github.io/elegy/guides/losses-and-metrics)`.
            optimizer: A `optax` optimizer instance. Optix is a very flexible library for defining
                optimization pipelines with things like learning rate schedules, this means that
                there is no need for a `LearningRateScheduler` callback in Elegy.
            run_eagerly: Settable attribute indicating whether the model should run eagerly.
                Running eagerly means that your model will be run step by step, like Python code, instead of
                using Jax's `jit` to. Your model might run slower, but it should become easier for you to debug
                it by stepping into individual layer calls.
            parameters: A `haiku.Params` structure with the weights of the model.
            states: A `haiku.State` structure with non-trainable parameters of the model.
            optimizer_state:  A `optax.OptState` structure with states of the optimizer.
            metrics_states: A `haiku.State` structure with the states of the metrics.
            initial_metrics_state: A `haiku.State` structure with the initial states of the metrics.
            seed: The initial random states of the model.
        """

        self.module = module
        self.loss_module = loss_modes.Losses(loss) if loss is not None else None
        self.metrics_module = metric_modes.Metrics(metrics) if metrics else None
        self._optimizer = optimizer if optimizer is not None else optax.adam(1e-3)
        self._rngs = PRNGSequence(seed)
        self._parameters = parameters
        self._states = states
        self._metrics_states = (
            metrics_states if self.metrics_module is not None else None
        )
        self.optimizer_state = optimizer_state
        self.initial_metrics_state = initial_metrics_state
        self.run_eagerly = run_eagerly

        utils.wraps(self.module)(self)

    @property
    def parameters(self) -> tp.Optional[types.Parameters]:
        return self._parameters

    @parameters.setter
    def parameters(self, values: types.Parameters):
        self.module.set_parameters(values)
        self._parameters = values

    @property
    def states(self) -> tp.Optional[types.States]:
        return self._states

    @states.setter
    def states(self, values: types.States):
        self.module.set_states(values)
        self._states = values

    @property
    def metrics_states(self) -> tp.Optional[types.States]:
        return self._metrics_states

    @metrics_states.setter
    def metrics_states(self, values: types.States):
        if self.metrics_module is not None:
            self.metrics_module.set_states(values)
            self._metrics_states = values
        else:
            raise ValueError("Cannot set metrics_states when metrics_module is None")

    def reset_metrics(self, hard: bool = False):

        if hard:
            self.metrics_module.reset()
            self._metrics_states = None
            self.initial_metrics_state = None
        elif self.initial_metrics_state is not None:
            self.metrics_states = self.initial_metrics_state

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def _maybe_initialize(
        self,
        mode: Mode,
        x: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple],
        y: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple, None],
        sample_weight: tp.Optional[jnp.ndarray],
        class_weight: tp.Optional[jnp.ndarray],
    ):

        # TODO(cgarciae): consider if maybe_jit can actually work else remove it
        # maybe_jit = jax.jit if not self.run_eagerly else lambda x: x
        maybe_jit = lambda x: x

        if self.parameters is None or self.states is None:
            x_args, x_kwargs = utils.get_input_args(x, training=True)

            self.parameters, self.states = maybe_jit(
                utils.inject_dependencies(self.module.init(rng=next(self._rngs)))
            )(*x_args, **x_kwargs)

        if mode == Mode.predict:
            return

        if self.metrics_module is not None and self.metrics_states is None:
            context: ApplyContext

            loss, (y_pred, context, logs) = maybe_jit(self._loss)(
                parameters=self.parameters,
                states=self.states,
                rng=next(self._rngs),
                x=x,
                y=y,
                sample_weight=sample_weight,
                class_weight=class_weight,
                training=True,
            )

            _, self.metrics_states = maybe_jit(
                self.metrics_module.init(rng=next(self._rngs))
            )(
                logs,
                x=x,
                y_true=y,
                y_pred=y_pred,
                sample_weight=sample_weight,
                class_weight=class_weight,
                training=False,
                parameters=self.parameters,
                states=self.states,
            )

            self.initial_metrics_state = self.metrics_states

        if mode == Mode.test:
            return

        if self.optimizer_state is None:
            self.optimizer_state = maybe_jit(self._optimizer.init)(self.parameters)

    def train_on_batch(
        self,
        x: tp.Union[np.ndarray, tp.Mapping[str, tp.Any], tp.Tuple],
        y: tp.Union[np.ndarray, tp.Mapping[str, tp.Any], tp.Tuple, None] = None,
        sample_weight: tp.Optional[np.ndarray] = None,
        class_weight: tp.Optional[np.ndarray] = None,
    ) -> tp.Dict[str, np.ndarray]:
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
        self._maybe_initialize(
            mode=Mode.train,
            x=x,
            y=y,
            sample_weight=sample_weight,
            class_weight=class_weight,
        )

        assert self.parameters is not None
        assert self.states is not None
        assert self.optimizer_state is not None

        (
            logs,
            self.parameters,
            self.states,
            self.optimizer_state,
            metrics_states,
        ) = self._update(
            x=x,
            y=y,
            sample_weight=sample_weight,
            class_weight=class_weight,
            parameters=self.parameters,
            states=self.states,
            optimizer_state=self.optimizer_state,
            metrics_states=self.metrics_states,
            rng=next(self._rngs),
        )

        if metrics_states is not None:
            self.metrics_states = metrics_states

        # logs = jax.tree_map(np.asarray, logs)

        return logs

    def _update(
        self,
        x: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple],
        y: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple, None],
        sample_weight: tp.Optional[jnp.ndarray],
        class_weight: tp.Optional[jnp.ndarray],
        parameters: tp.Dict,
        states: tp.Dict,
        optimizer_state: optax.OptState,
        metrics_states: tp.Optional[tp.Dict],
        rng: jnp.ndarray,
    ) -> tp.Tuple[
        tp.Dict[str, jnp.ndarray],
        tp.Dict,
        tp.Dict,
        optax.OptState,
        tp.Optional[tp.Dict],
    ]:
        update_fn = self._update_no_jit if self.run_eagerly else self._update_jit

        return update_fn(
            x=x,
            y=y,
            sample_weight=sample_weight,
            class_weight=class_weight,
            parameters=parameters,
            states=states,
            optimizer_state=optimizer_state,
            metrics_states=metrics_states,
            rng=rng,
        )

    def _update_no_jit(
        self,
        x: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple],
        y: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple, None],
        sample_weight: tp.Optional[jnp.ndarray],
        class_weight: tp.Optional[jnp.ndarray],
        parameters: tp.Dict,
        states: tp.Dict,
        optimizer_state: optax.OptState,
        metrics_states: tp.Optional[tp.Dict],
        rng: jnp.ndarray,
    ) -> tp.Tuple[
        tp.Dict[str, jnp.ndarray],
        tp.Dict,
        tp.Dict,
        optax.OptState,
        tp.Optional[tp.Dict],
    ]:
        net_rng, metrics_rng = jax.random.split(rng, num=2)

        (loss, (y_pred, context, logs)), grads = jax.value_and_grad(
            self._loss, has_aux=True
        )(
            parameters,
            states=states,
            rng=net_rng,
            x=x,
            y=y,
            sample_weight=sample_weight,
            class_weight=class_weight,
            training=True,
        )

        states = context.states

        updates, optimizer_state = self._optimizer.update(
            grads, optimizer_state, parameters
        )
        parameters = optax.apply_updates(parameters, updates)

        if self.metrics_module is not None:
            logs, metrics_context = utils.inject_dependencies(
                self.metrics_module.apply(states=metrics_states, rng=metrics_rng)
            )(
                logs,
                x=x,
                y_true=y,
                y_pred=y_pred,
                sample_weight=sample_weight,
                class_weight=class_weight,
                training=True,
                parameters=parameters,
                states=states,
            )
            metrics_states = metrics_context.states

        return logs, parameters, states, optimizer_state, metrics_states

    _update_jit = jax.jit(_update_no_jit, static_argnums=(0,))

    def _loss(
        self,
        parameters: tp.Dict,
        states: tp.Dict,
        rng: jnp.ndarray,
        x: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple],
        y: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple, None],
        sample_weight: tp.Optional[jnp.ndarray],
        class_weight: tp.Optional[jnp.ndarray],
        training: bool,
    ):
        rng, loss_rng = jax.random.split(rng, num=2)

        y_pred, context = self._predict_no_jit(
            training=training,
            get_summaries=False,
            x=x,
            parameters=parameters,
            states=states,
            rng=rng,
        )

        states = context.states

        logs, loss_context = (
            self.loss_module.apply(rng=loss_rng)(
                x=x,
                y_true=y,
                y_pred=y_pred,
                sample_weight=sample_weight,
                class_weight=class_weight,
                training=training,
                parameters=parameters,
                states=states,
            )
            if self.loss_module is not None
            else ({}, None)
        )

        # calculate total loss
        loss = sum(logs.values()) + sum(context.losses.values())

        # add losses and metrics from hooks
        if loss_context is not None:
            logs.update(
                {
                    name.replace("losses/", ""): value
                    for name, value in loss_context.metrics.items()
                }
            )

        logs.update(context.losses)
        logs.update(context.metrics)

        # set total loss
        logs["loss"] = loss

        return loss, (y_pred, context, logs)

    def fit(
        self,
        x: tp.Union[
            np.ndarray,
            tp.Mapping[str, np.ndarray],
            tp.Tuple[np.ndarray],
            tp.Iterable,
            None,
        ] = None,
        y: tp.Union[
            np.ndarray,
            tp.Mapping[str, np.ndarray],
            tp.Tuple[np.ndarray],
            None,
        ] = None,
        batch_size: tp.Optional[int] = None,
        epochs: int = 1,
        verbose: int = 1,
        callbacks: tp.Union[tp.List[Callback], CallbackList, None] = None,
        validation_split: float = 0.0,
        validation_data: tp.Union[tp.Tuple, tp.Iterable, None] = None,
        shuffle: bool = True,
        class_weight: tp.Optional[tp.Mapping[str, float]] = None,
        sample_weight: tp.Optional[np.ndarray] = None,
        initial_epoch: int = 0,
        steps_per_epoch: tp.Optional[int] = None,
        validation_steps: tp.Optional[int] = None,
        validation_batch_size: tp.Optional[int] = None,
        validation_freq: int = 1,
    ) -> History:
        """
        Trains the model for a fixed number of epochs (iterations on a dataset).

        Arguments:
            x: Input data. It could be:

                - A Numpy or Jax array (or array-like), or a list of arrays
                    (in case the model has multiple inputs).
                - A dict mapping input names to the corresponding arrays,
                    if the model has named inputs.
                - A generator returning `(inputs,)`, `(inputs, targets)`
                    or `(inputs, targets, sample_weights)`.

                A more detailed description of unpacking behavior for generator type
                is given below.
            y: Target data. Like the input data `x`,
                it could be either Numpy or Jax array(s).
                It should be consistent with `x`. If `x` is a generator,
                `y` should not be specified (since targets will be obtained from `x`).
            batch_size: Integer or `None`.
                Number of samples per gradient update.
                If unspecified, `batch_size` will default to 32.
                Do not specify the `batch_size` if your data is in the
                form of generator (since they generate batches).
            epochs: Integer. Number of epochs to train the model.
                An epoch is an iteration over the entire `x` and `y`
                data provided.
                Note that in conjunction with `initial_epoch`,
                `epochs` is to be understood as "final epoch".
                The model is not trained for a number of iterations
                given by `epochs`, but merely until the epoch
                of index `epochs` is reached.
            verbose: 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
                Note that the progress bar is not particularly useful when
                logged to a file, so verbose=2 is recommended when not running
                interactively (eg, in a production environment).
            callbacks: List of [elegy.callbacks.callback.Callback][] instances.
                List of callbacks to apply during training.
            validation_split: Float between 0 and 1.
                Fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate
                the loss and any model metrics
                on this data at the end of each epoch.
                The validation data is selected from the last samples
                in the `x` and `y` data provided, before shuffling. This argument is
                not supported when `x` is a generator.
            validation_data: Data on which to evaluate
                the loss and any model metrics at the end of each epoch.
                The model will not be trained on this data.
                `validation_data` will override `validation_split`.
                `validation_data` could be:

                - tuple `(x_val, y_val)` of Numpy/Jax arrays, list of arrays or mappings
                - tuple `(x_val, y_val, val_sample_weights)` of Numpy/Jax arrays, list
                of arrays or mappings
                - generator

                For the first two cases, `batch_size` must be provided.
                For the last case, `validation_steps` should be provided, and should
                follow the same convention for yielding data as `x`.
                Note that `validation_data` does not support all the data types that
                are supported in `x`, eg, dict.
            shuffle: Boolean (whether to shuffle the training data
                before each epoch). This argument is ignored
                when `x` is a generator. Has no effect when `steps_per_epoch` is not `None`.
            class_weight: Optional dictionary mapping class indices (integers)
                to a weight (float) value, used for weighting the loss function
                (during training only).
                This can be useful to tell the model to
                "pay more attention" to samples from
                an under-represented class.
            sample_weight: Optional Numpy/Jax array of weights for
                the training samples, used for weighting the loss function
                (during training only). You can either pass a flat (1D)
                Numpy array with the same length as the input samples
                (1:1 mapping between weights and samples). This argument is not
                supported when `x` is generator, instead provide the sample_weights
                as the third element of `x`.
            initial_epoch: Integer.
                Epoch at which to start training
                (useful for resuming a previous training run).
            steps_per_epoch: Integer or `None`.
                Total number of steps (batches of samples)
                before declaring one epoch finished and starting the
                next epoch. When training with input arrays such as
                jax data arrays, the default `None` is equal to
                the number of samples in your dataset divided by
                the batch size, or 1 if that cannot be determined.
                When passing a generator, you must specify the
                `steps_per_epoch` argument. This argument is not supported with
                array inputs.
            validation_steps: Only relevant if `validation_data` is provided and
                is a generator. Total number of steps (batches of
                samples) to draw before stopping when performing validation
                at the end of every epoch. If 'validation_steps' is None, validation
                will run until the `validation_data` dataset is exhausted. In the
                case of an infinitely repeated dataset, it will run into an
                infinite loop. If 'validation_steps' is specified and only part of
                the dataset will be consumed, the evaluation will start from the
                beginning of the dataset at each epoch. This ensures that the same
                validation samples are used every time.
            validation_batch_size: Integer or `None`.
                Number of samples per validation batch.
                If unspecified, will default to `batch_size`.
                Do not specify the `validation_batch_size` if your data is in the
                form of generators (since they generate batches).
            validation_freq: Only relevant if validation data is provided. Integer
                or `collections_abc.Container` instance (e.g. list, tuple, etc.).
                If an integer, specifies how many training epochs to run before a
                new validation run is performed, e.g. `validation_freq=2` runs
                validation every 2 epochs. If a Container, specifies the epochs on
                which to run validation, e.g. `validation_freq=[1, 2, 10]` runs
                validation at the end of the 1st, 2nd, and 10th epochs.

        Unpacking behavior for iterator-like inputs:

        A common pattern is to pass a generator, which will in fact
        yield not only features (x) but optionally targets (y) and sample weights.
        Elegy requires that the output of such iterator-likes be unambiguous. The
        iterator should return a tuple of length 1, 2, or 3, where the optional
        second and third elements will be used for y and sample_weight
        respectively. Any other type provided will be wrapped in a length one
        tuple, effectively treating everything as 'x'. When yielding dicts, they
        should still adhere to the top-level tuple structure.
        e.g. `({"x0": x0, "x1": x1}, y)`. Elegy will not attempt to separate
        features, targets, and weights from the keys of a single dict.

        A notable unsupported data type is the namedtuple. The reason is that
        it behaves like both an ordered datatype (tuple) and a mapping
        datatype (dict). So given a namedtuple of the form:
            `namedtuple("example_tuple", ["y", "x"])`
        it is ambiguous whether to reverse the order of the elements when
        interpreting the value. Even worse is a tuple of the form:
            `namedtuple("other_tuple", ["x", "y", "z"])`
        where it is unclear if the tuple was intended to be unpacked into x, y,
        and sample_weight or passed through as a single element to `x`. As a
        result the data processing code will simply raise a ValueError if it
        encounters a namedtuple. (Along with instructions to remedy the issue.)

        Returns:
            A `History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).
        Raises:
            ValueError: In case of mismatch between the provided input data
                and what the model expects.
        """
        if x is None:
            x = {}

        if validation_split:
            # Create the validation data using the training data. Only supported for
            # `Jax Numpy` and `NumPy` input.
            (x, y, sample_weight), validation_data = train_validation_split(
                (x, y, sample_weight), validation_split=validation_split, shuffle=False
            )

        self.stop_training = False
        data_handler = DataHandler(
            x=x,
            y=y,
            sample_weight=sample_weight,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            initial_epoch=initial_epoch,
            epochs=epochs,
            shuffle=shuffle,
            class_weight=class_weight,
        )
        # Container that configures and calls `tf.keras.Callback`s.
        if not isinstance(callbacks, CallbackList):
            callbacks = CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                model=self,
                verbose=verbose,
                epochs=epochs,
                steps=data_handler.inferred_steps,
            )

        callbacks.on_train_begin()
        # data_handler._initial_epoch = (  # pylint: disable=protected-access
        #     self._maybe_load_initial_epoch_from_ckpt(initial_epoch))

        for epoch, iterator in data_handler.enumerate_epochs():
            self.reset_metrics()
            callbacks.on_epoch_begin(epoch)
            logs = {}
            with data_handler.catch_stop_iteration():
                for step in data_handler.steps():
                    callbacks.on_train_batch_begin(step)
                    batch = next(iterator)
                    # sample_weight = batch[2] if len(batch) == 3 else None
                    x_batch, y_batch, sample_weight = unpack_x_y_sample_weight(batch)

                    tmp_logs = self.train_on_batch(
                        x=x_batch,
                        y=y_batch,
                        sample_weight=sample_weight,
                        class_weight=class_weight,
                    )
                    tmp_logs.update({"size": data_handler.batch_size})
                    # print(epoch, step, tmp_logs["accuracy"], batch[0].shape)

                    logs = tmp_logs
                    callbacks.on_train_batch_end(step, logs)

            epoch_logs = copy(logs)
            epoch_logs.update({"size": data_handler.batch_size})

            # Run validation.
            if validation_data and self._should_eval(epoch, validation_freq):
                val_x, val_y, val_sample_weight = unpack_x_y_sample_weight(
                    validation_data
                )
                val_logs = self.evaluate(
                    x=val_x,
                    y=val_y,
                    sample_weight=val_sample_weight,
                    batch_size=validation_batch_size or batch_size,
                    steps=validation_steps,
                    callbacks=callbacks,
                    # return_dict=True,
                )
                val_logs = {"val_" + name: val for name, val in val_logs.items()}
                epoch_logs.update(val_logs)

            callbacks.on_epoch_end(epoch, epoch_logs)
            # print(
            #     f"epoch: {epoch} - "
            #     + " - ".join(f"{key}: {value:.3f}" for key, value in epoch_logs.items())
            # )
            if self.stop_training:
                break

        callbacks.on_train_end()

        return self.history

    def evaluate(
        self,
        x: tp.Union[
            np.ndarray,
            tp.Mapping[str, np.ndarray],
            tp.Tuple[np.ndarray],
            tp.Iterable,
        ],
        y: tp.Union[
            jnp.ndarray,
            np.ndarray,
            tp.Mapping[str, np.ndarray],
            tp.Tuple[np.ndarray],
            None,
        ] = None,
        verbose: int = 1,
        batch_size: tp.Optional[int] = None,
        sample_weight: tp.Optional[np.ndarray] = None,
        steps: tp.Optional[int] = None,
        callbacks: tp.Union[tp.List[Callback], CallbackList, None] = None,
    ) -> tp.Dict[str, np.ndarray]:
        """Returns the loss value & metrics values for the model in test mode.
        Computation is done in batches.

        Arguments:
            x: Input data. It could be:

                - A Numpy or Jax array (or array-like), or a list of arrays
                    (in case the model has multiple inputs).
                - A dict mapping input names to the corresponding arrays,
                    if the model has named inputs.
                - A generator returning `(inputs,)`, `(inputs, targets)`
                    or `(inputs, targets, sample_weights)`.

                A more detailed description of
                unpacking behavior for iterator types generator
                is given in the `Unpacking behavior for iterator-like inputs` section
                of `Model.fit`.
            y: Target data. Like the input data `x`,
                it could be either Numpy or Jax array(s).
                It should be consistent with `x`. If `x` is a generator,
                `y` should not be specified (since targets will be obtained from `x`).
            verbose: 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
            batch_size: Integer or `None`.
                Number of samples per gradient update.
                If unspecified, `batch_size` will default to 32.
                Do not specify the `batch_size` if your data is in the
                form of generator (since they generate batches).
            sample_weight: Optional Numpy/Jax array of weights for
                the training samples, used for weighting the loss function
                (during training only). You can either pass a flat (1D)
                Numpy array with the same length as the input samples
                (1:1 mapping between weights and samples). This argument is not
                supported when `x` is generator, instead provide the sample_weights
                as the third element of `x`.
            steps: Integer or `None`. Total number of steps (batches of samples)
                before declaring the evaluation round finished. Ignored with the
                default value of `None`. This
                argument is not supported with array inputs.
            callbacks: List of [elegy.callbacks.callback.Callback][] instances.
                List of callbacks to apply during training.

        See the discussion of `Unpacking behavior for iterator-like inputs` for
         [`Model.fit`][elegy.model.Model.fit].

        Returns:
            A dictionary for mapping the losses and metrics names to the values obtained.
        Raises:
            ValueError: in case of invalid arguments.
        """

        data_handler = DataHandler(
            x=x,
            y=y,
            sample_weight=sample_weight,
            batch_size=batch_size,
            steps_per_epoch=steps,
            initial_epoch=0,
            epochs=1,
            shuffle=False,
            training=False,
        )

        # Container that configures and calls `tf.keras.Callback`s.
        if not isinstance(callbacks, CallbackList):
            callbacks = CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                model=self,
                verbose=verbose,
                epochs=1,
                steps=data_handler.inferred_steps,
            )

        callbacks.on_test_begin()

        logs = {}
        for _, iterator in data_handler.enumerate_epochs():
            self.reset_metrics()
            with data_handler.catch_stop_iteration():
                for step in data_handler.steps():
                    callbacks.on_test_batch_begin(step)
                    batch = next(iterator)
                    x_batch, y_batch, sample_weight = unpack_x_y_sample_weight(batch)

                    tmp_logs = self.test_on_batch(
                        x=x_batch,
                        y=y_batch,
                        sample_weight=sample_weight,
                    )
                    tmp_logs.update({"size": data_handler.batch_size})
                    logs = tmp_logs
                    callbacks.on_test_batch_end(step, logs)

        callbacks.on_test_end()

        return logs

    def predict(
        self,
        x: tp.Union[
            np.ndarray,
            tp.Mapping[str, np.ndarray],
            tp.Tuple[np.ndarray],
            tp.Iterable,
        ],
        verbose: int = 0,
        batch_size: tp.Optional[int] = None,
        steps: tp.Optional[int] = None,
        callbacks: tp.Union[tp.List[Callback], CallbackList, None] = None,
    ) -> np.ndarray:
        """Generates output predictions for the input samples.
        Computation is done in batches.

        Arguments:
            x: Input data. It could be:

                - A Numpy or Jax array (or array-like), or a list of arrays
                    (in case the model has multiple inputs).
                - A dict mapping input names to the corresponding arrays,
                    if the model has named inputs.
                - A generator returning `(inputs,)`, `(inputs, targets)`
                    or `(inputs, targets, sample_weights)`.

                A more detailed description of
                unpacking behavior for iterator types generator
                is given in the `Unpacking behavior for iterator-like inputs` section
                of `Model.fit`.
            batch_size: Integer or `None`.
                Number of samples per batch.
                If unspecified, `batch_size` will default to 32.
                Do not specify the `batch_size` if your data is in the
                form of generators (since they generate batches).
            verbose: Verbosity mode, 0 or 1.
            steps: Total number of steps (batches of samples)
                before declaring the prediction round finished.
                Ignored with the default value of `None`.
            callbacks: List of [elegy.callbacks.callback.Callback][] instances.
                List of callbacks to apply during training.

        See the discussion of `Unpacking behavior for iterator-like inputs` for
        [`Model.fit`][elegy.model.Model.fit].
        Note that Model.predict uses the same interpretation rules as
        `Model.fit` and `Model.evaluate`, so inputs must be unambiguous for all
        three methods.
        Returns:
            Numpy array(s) of predictions.
        Raises:
            ValueError: In case of mismatch between the provided
                input data and the model's expectations,
                or in case a stateful model receives a number of samples
                that is not a multiple of the batch size.
        """

        outputs = None

        data_handler = DataHandler(
            x=x,
            batch_size=batch_size,
            steps_per_epoch=steps,
            initial_epoch=0,
            epochs=1,
            shuffle=False,
        )

        # Container that configures and calls `tf.keras.Callback`s.
        if not isinstance(callbacks, CallbackList):
            callbacks = CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                model=self,
                verbose=verbose,
                epochs=1,
                steps=data_handler.inferred_steps,
            )

        callbacks.on_predict_begin()

        for _, iterator in data_handler.enumerate_epochs():
            self.reset_metrics()
            with data_handler.catch_stop_iteration():
                for step in data_handler.steps():
                    callbacks.on_predict_batch_begin(step)
                    batch = next(iterator)
                    tmp_batch_outputs = self.predict_on_batch(x=batch[0])
                    batch_outputs = tmp_batch_outputs

                    if outputs is None:
                        outputs = map_structure(
                            lambda batch_output: [batch_output], batch_outputs
                        )
                    else:

                        outputs = map_structure(
                            map_append,
                            outputs,
                            batch_outputs,
                        )

                    callbacks.on_predict_batch_end(
                        step,
                        {"outputs": batch_outputs, "size": data_handler.batch_size},
                    )

        callbacks.on_predict_end()

        all_outputs = map_structure(jnp.concatenate, outputs)

        return all_outputs

    def _should_eval(self, epoch, validation_freq):
        epoch = epoch + 1  # one-index the user-facing epoch.
        if isinstance(validation_freq, int):
            return epoch % validation_freq == 0
        elif isinstance(validation_freq, list):
            return epoch in validation_freq
        else:
            raise ValueError("Expected `validation_freq` to be a list or int.")

    def test_on_batch(
        self,
        x: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple],
        y: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple, None] = None,
        sample_weight: tp.Optional[jnp.ndarray] = None,
        class_weight: tp.Optional[jnp.ndarray] = None,
    ) -> tp.Dict[str, jnp.ndarray]:
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
        self._maybe_initialize(
            mode=Mode.test,
            x=x,
            y=y,
            sample_weight=sample_weight,
            class_weight=class_weight,
        )

        assert self.parameters is not None
        assert self.states is not None

        (logs, metrics_states) = self._test(
            x=x,
            y=y,
            sample_weight=sample_weight,
            class_weight=class_weight,
            parameters=self.parameters,
            states=self.states,
            metrics_states=self.metrics_states,
            rng=next(self._rngs),
        )

        if metrics_states is not None:
            self.metrics_states = metrics_states

        # logs = jax.tree_map(np.asarray, logs)

        return logs

    def _test(
        self,
        x: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple],
        y: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple, None],
        sample_weight: tp.Optional[jnp.ndarray],
        class_weight: tp.Optional[jnp.ndarray],
        parameters: tp.Dict,
        states: tp.Dict,
        metrics_states: tp.Optional[tp.Dict],
        rng: jnp.ndarray,
    ) -> tp.Tuple[tp.Dict[str, jnp.ndarray], tp.Optional[tp.Dict],]:

        test_fn = self._test_no_jit if self.run_eagerly else self._test_jit

        return test_fn(
            x=x,
            y=y,
            sample_weight=sample_weight,
            class_weight=class_weight,
            parameters=parameters,
            states=states,
            metrics_states=metrics_states,
            rng=rng,
        )

    def _test_no_jit(
        self,
        x: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple],
        y: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple, None],
        sample_weight: tp.Optional[jnp.ndarray],
        class_weight: tp.Optional[jnp.ndarray],
        parameters: tp.Dict,
        states: tp.Dict,
        metrics_states: tp.Optional[tp.Dict],
        rng: jnp.ndarray,
    ) -> tp.Tuple[tp.Dict[str, jnp.ndarray], tp.Optional[tp.Dict],]:
        net_rng, metrics_rng = jax.random.split(rng, num=2)

        loss, (y_pred, context, logs) = self._loss(
            parameters,
            states=states,
            rng=net_rng,
            x=x,
            y=y,
            sample_weight=sample_weight,
            class_weight=class_weight,
            training=False,
        )

        if self.metrics_module is not None:
            logs, metrics_context = utils.inject_dependencies(
                self.metrics_module.apply(
                    states=metrics_states, rng=metrics_rng, training=False
                ),
            )(
                logs,
                x=x,
                y_true=y,
                y_pred=y_pred,
                sample_weight=sample_weight,
                class_weight=class_weight,
                training=False,
                __params=parameters,
                __state=states,
            )
            metrics_states = metrics_context.states

        return logs, metrics_states

    _test_jit = jax.jit(_test_no_jit, static_argnums=(0,))
    # ----------------------------------------------------------------
    # predict
    # ----------------------------------------------------------------

    def predict_on_batch(
        self, x: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple]
    ) -> tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple]:
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
        self._maybe_initialize(
            mode=Mode.predict, x=x, y=None, sample_weight=None, class_weight=None
        )

        assert self.parameters is not None
        assert self.states is not None

        y_pred, context = self._predict(
            False,  # training
            False,  # get_summaries
            x=x,
            parameters=self.parameters,
            states=self.states,
            rng=next(self._rngs),
        )

        return y_pred

    def _predict(
        self,
        training: bool,
        get_summaries: bool,
        x: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple],
        parameters: tp.Dict,
        states: tp.Dict,
        rng: jnp.ndarray,
    ) -> tp.Tuple[tp.Any, "ApplyContext"]:

        predict_fn = self._predict_no_jit if self.run_eagerly else self._predict_jit

        return predict_fn(
            training,
            get_summaries,
            x=x,
            parameters=parameters,
            states=states,
            rng=rng,
        )

    def _predict_no_jit(
        self,
        training: bool,
        get_summaries: bool,
        x: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple],
        parameters: tp.Dict,
        states: tp.Dict,
        rng: jnp.ndarray,
    ) -> tp.Tuple[tp.Any, ApplyContext]:

        x_args, x_kwargs = utils.get_input_args(x, training=training)

        return utils.inject_dependencies(
            self.module.apply(
                parameters=parameters,
                states=states,
                rng=rng,
                get_summaries=get_summaries,
                training=training,
            )
        )(*x_args, **x_kwargs)

    _predict_jit = jax.jit(_predict_no_jit, static_argnums=(0, 1, 2))

    @property
    def seed(self) -> np.ndarray:
        """
        Current random states of the model.
        """
        return self._rngs.key

    @seed.setter
    def seed(self, seed: tp.Union[np.ndarray, int]):
        self._rngs = PRNGSequence(seed)

    @property
    def full_state(self) -> tp.Dict:
        """"""

        states: tp.Dict = {"seed": self.seed}

        if self.parameters is not None:
            states["parameters"] = self.parameters

        if self.states is not None:
            states["states"] = self.states

        if self.metrics_states is not None:
            states["metrics_states"] = self.metrics_states

        if self.initial_metrics_state is not None:
            states["initial_metrics_state"] = self.initial_metrics_state

        if self.optimizer_state is not None:
            states["optimizer_state"] = self.optimizer_state

        return states

    @full_state.setter
    def full_state(self, states: tp.Dict):
        self.seed = states["seed"]

        if "parameters" in states:
            self.parameters = states["parameters"]

        if "states" in states:
            self.states = states["states"]

        if "metrics_states" in states:
            self.metrics_states = states["metrics_states"]

        if "initial_metrics_state" in states:
            self.initial_metrics_state = states["initial_metrics_state"]

        if "optimizer_state" in states:
            self.optimizer_state = states["optimizer_state"]

    def reset(self):
        self.module.reset()
        self.metrics_module.reset()
        self.initial_metrics_state = None
        self.optimizer_state = None

    def save(self, path: tp.Union[str, Path], include_optimizer: bool = True) -> None:
        """
        Saves the model to disk.

        It creates a directory that includes:

        - The `Model` object instance serialized with `pickle` as
            as `{path}/model.pkl`, this allows you to re-instantiate
            the model later.
        - The model parameters + states serialized into HDF5 as `{path}/parameters.h5`.
        - The states of the optimizer serialized with `pickle` as
            as `{path}/optimizer_state.pkl`, allowing to resume training
            exactly where you left off. We hope to use HDF5 in the future
            but `optax` states is incompatible with `deepdish`.

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
            include_optimizer: If True, save optimizer's states together.
        """
        if isinstance(path, str):
            path = Path(path)

        path.mkdir(parents=True, exist_ok=True)

        states = self.full_state

        original_state = copy(states)

        states.pop("metrics_states", None)
        states.pop("initial_metrics_state", None)

        optimizer_state = states.pop("optimizer_state", None)

        deepdish.io.save(path / "parameters.h5", states)

        if include_optimizer and optimizer_state is not None:
            with open(path / "optimizer_state.pkl", "wb") as f:
                pickle.dump(optimizer_state, f)

        # getting pickle errors
        self.reset()

        try:
            path = path / "model.pkl"
            with open(path, "wb") as f:
                cloudpickle.dump(self, f)
        except BaseException as e:
            print(f"Error occurred saving the model object at {path}\nContinuing....")

        self.full_state = original_state

    def load(self, path: tp.Union[str, Path]) -> None:
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
            path = Path(path)

        states: tp.Dict = deepdish.io.load(path / "parameters.h5")

        optimizer_state_path = path / "optimizer_state.pkl"

        if optimizer_state_path.exists():
            with open(optimizer_state_path, "rb") as f:
                states["optimizer_state"] = pickle.load(f)

        self.full_state = states

    def summary(
        self, x, depth: int = 2, tablefmt: str = "fancy_grid", **tablulate_kwargs
    ):
        """
        Prints a summary of the network.

        Arguments:
            x: A sample of inputs to the network.
            depth: The level number of nested level which will be showed.
                Information about summaries from modules deeper than `depth`
                will be aggregated together.
            tablefmt: A string represeting the style of the table generated by
                `tabulate`. See
                [python-tabulate](https://github.com/astanin/python-tabulate)
                for more options.
            tablulate_kwargs: Additional keyword arguments passed to `tabulate`.
                See [python-tabulate](https://github.com/astanin/python-tabulate)
                for more options.
        """
        self._maybe_initialize(
            mode=Mode.predict,
            x=x,
            y=None,
            sample_weight=None,
            class_weight=None,
        )

        assert self.parameters is not None
        assert self.states is not None

        y_pred, context = self._predict(
            training=False,
            get_summaries=True,
            x=x,
            parameters=self.parameters,
            states=self.states,
            rng=next(self._rngs),
        )

        def format_output(outputs) -> str:
            file = StringIO()
            outputs = jax.tree_map(lambda x: f"{x.shape}{{pad}}  {x.dtype}", outputs)
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

        table: tp.List = [["Inputs", format_output(x), "0", "0"]]

        for module, base_name, value in context.summaries:
            base_name_parts = base_name.split("/")[1:]
            module_depth = len(base_name_parts)

            if module_depth > depth:
                continue

            include_submodules = module_depth == depth

            params_count = (
                module.parameters_size(include_submodules) if module is not None else 0
            )
            params_size = (
                module.parameters_bytes(include_submodules) if module is not None else 0
            )
            states_count = (
                module.states_size(include_submodules) if module is not None else 0
            )
            states_size = (
                module.states_bytes(include_submodules) if module is not None else 0
            )
            class_name = f"({module.__class__.__name__})" if module is not None else ""

            base_name = "/".join(base_name_parts)

            if not base_name:
                base_name = "Outputs"

            table.append(
                [
                    f"{base_name}{{pad}}  {class_name}",
                    format_output(value),
                    f"{params_count:,}{{pad}}    {format_size(params_size)}"
                    if params_count > 0
                    else "0",
                    f"{states_count:,}{{pad}}    {format_size(states_size)}"
                    if states_count > 0
                    else "0",
                ]
            )

        # add papdding
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

        print(
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
        )

        params_count = self.module.parameters_size()
        params_size = self.module.parameters_bytes()
        states_count = self.module.states_size()
        states_size = self.module.states_bytes()
        total_count = params_count + states_count
        total_size = params_size + states_size

        print(
            tabulate(
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


def load(path: tp.Union[str, Path]) -> Model:
    """
    Loads a model from disk.

    This function will restore both the model architecture,
    that is, its `Model` class instance, along with all of its
    parameters, states, and optimizer states.

    Example:

    ```python
    import elegy

    model.save('my_model')  # creates folder at 'my_model'
    del model  # deletes the existing model

    # returns a model identical to the previous one
    model = elegy.model.load('my_model')
    ```

    Arguments:
        path: path to a saved model's directory.

    Raises:
        OSError: in case the model was not found or could not be
            loaded from disk successfully.
    """
    if isinstance(path, str):
        path = Path(path)

    with open(path / "model.pkl", "rb") as f:
        try:
            model: Model = pickle.load(f)
        except BaseException as e:
            raise OSError(f"Could not load the model. Got exception: {e}")

    model.load(path)

    return model
