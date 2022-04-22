import pickle
import typing as tp
from copy import copy
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import treex as tx
from optax import GradientTransformation

from elegy import data, types, utils
from elegy.callbacks import Callback, CallbackList, History
from elegy.callbacks.sigint import SigIntMode
from elegy.data import utils as data_utils
from elegy.modules.core_module import CoreModule

M = tp.TypeVar("M", bound="Model")
U = tp.TypeVar("U", bound="tx.Module")


try:
    import haiku as hk

    TransformedWithState = hk.TransformedWithState
    HaikuModule = tx.HaikuModule
except (ImportError, ModuleNotFoundError):
    hk = None
    TransformedWithState = type(None)
    HaikuModule = tp.cast(tp.Any, None)


class Model:
    """
    Model provides an Estimator-like API similar to Keras.
    """

    module: CoreModule
    seed: tp.Union[int, jnp.ndarray]
    history: tp.Optional[History]
    stop_training: bool

    def __init__(
        self,
        module: CoreModule,
        loss: tp.Any = None,
        metrics: tp.Any = None,
        optimizer: tp.Optional[tp.Union[tx.Optimizer, GradientTransformation]] = None,
        seed: int = 42,
        eager: bool = False,
    ):
        """
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
            eager: Settable attribute indicating whether the model should run eagerly.
                Running eagerly means that your model will be run step by step, like Python code, instead of
                using Jax's `jit` to. Your model might run slower, but it should become easier for you to debug
                it by stepping into individual layer calls.
        """
        self.seed = seed
        self.module = module
        self.history = None
        self.stop_training = False

        # TODO: CoreModule should have a method to set this
        # self.optimizer = (
        #     tx.Optimizer(optimizer)
        #     if isinstance(optimizer, GradientTransformation)
        #     else optimizer
        # )
        # self.loss_and_logs = None

        # self._losses_and_metrics = tx.Hashable((loss, metrics))

    def __call__(self, *args, **kwargs) -> tp.Any:
        return self.module(*args, **kwargs)

    # ----------------------------------------------------------------
    # Model-only methods
    # ----------------------------------------------------------------

    def reset_step(self):
        self.module = self.module.reset_step()

    def init_on_batch(self, batch: tp.Any):

        key = tx.Key(self.seed)

        module = self.module.init_step(key, batch)

        if not isinstance(module, type(self.module)):
            raise ValueError(
                f"`CoreModule.init_on_batch` must return an instance of {type(self.module)}, got {type(module)}."
            )

        self.module = module.mark_initialized()

    def _make_predict_step(self, batch: tp.Any, batch_idx: int) -> tp.Any:

        if not self.module.initialized:
            self.init_on_batch(batch)

        # do this after init_on_batch()
        batch = self.module.distributed_strategy.lift_data(batch)

        outputs, module = self.module.predict_step(batch, batch_idx)

        if not isinstance(module, type(self.module)):
            raise ValueError(
                f"'CoreModule.predict_on_batch' must return an instance of {type(self.module)}, got {type(module)}."
            )

        self.module = module

        return outputs

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
        raise NotImplementedError()

    def _make_test_step(
        self,
        batch: tp.Any,
        batch_idx: int,
    ) -> types.Logs:

        if not self.module.initialized:
            self.init_on_batch(batch)

        # do this after init_on_batch()
        batch = self.module.distributed_strategy.lift_data(batch)

        logs, module = self.module.test_step(batch, batch_idx)

        if not isinstance(module, type(self.module)):
            raise ValueError(
                f"'CoreModule.test_on_batch' must return an instance of {type(self.module)}, got: {type(module)}"
            )

        self.module = module

        return logs

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
        raise NotImplementedError()

    def _make_train_step(
        self,
        batch: tp.Any,
        batch_idx: int,
        epoch_idx: int,
    ) -> types.Logs:

        if not self.module.initialized:
            self.init_on_batch(batch)

        # do this after init_on_batch()
        batch = self.module.distributed_strategy.lift_data(batch)

        logs, module = self.module.train_step(batch, batch_idx, epoch_idx)

        if not isinstance(module, type(self.module)):
            raise ValueError(
                f"`CoreModule.train_on_batch` must return an instance of {type(self.module)}, got {type(module)}."
            )

        self.module = module

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
        raise NotImplementedError()

    def predict(
        self,
        x: tp.Optional[tp.Any] = None,
        verbose: int = 0,
        batch_size: tp.Optional[int] = 32,
        steps: tp.Optional[int] = None,
        callbacks: tp.Union[tp.List[Callback], CallbackList, None] = None,
        drop_remaining: bool = False,
    ) -> tp.Any:
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
        [`Model.fit`][elegy.model.model.Model.fit].
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
        if batch_size is not None:
            batch_size = self.module.distributed_strategy.lift_batch_size(batch_size)

        if x is None:
            x = {}

        outputs = None

        data_handler = data.DataHandler(
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
                sigint_mode=SigIntMode.PREDICT,
                add_module=True,
                model=self,
                verbose=verbose,
                epochs=1,
                steps=data_handler.inferred_steps,
            )

        callbacks.on_predict_begin()

        for _, iterator in data_handler.enumerate_epochs():
            self.reset_step()
            with data_handler.catch_stop_iteration():
                for step in data_handler.steps():
                    callbacks.on_predict_batch_begin(step)
                    batch = next(iterator)

                    if drop_remaining and not data_utils.has_batch_size(
                        batch, data_handler.batch_size
                    ):
                        continue

                    if isinstance(batch, (tuple, list)) and len(batch) == 1:
                        batch = batch[0]

                    batch_outputs = self._make_predict_step(batch, step)

                    if outputs is None:
                        outputs = data.map_structure(
                            lambda batch_output: [batch_output], batch_outputs
                        )
                    else:

                        outputs = data.map_structure(
                            data.map_append,
                            outputs,
                            batch_outputs,
                        )

                    callbacks.on_predict_batch_end(
                        step,
                        {"outputs": batch_outputs, "size": data_handler.batch_size},
                    )

                    if self.stop_training:
                        break
                if self.stop_training:
                    break

        callbacks.on_predict_end()

        all_outputs = data.map_structure(jnp.concatenate, outputs)

        return all_outputs

    def evaluate(
        self,
        inputs: tp.Optional[tp.Any] = None,
        labels: tp.Optional[tp.Any] = None,
        verbose: int = 1,
        batch_size: tp.Optional[int] = 32,
        sample_weight: tp.Optional[np.ndarray] = None,
        class_weight: tp.Optional[tp.Mapping[str, float]] = None,
        steps: tp.Optional[int] = None,
        callbacks: tp.Union[tp.List[Callback], CallbackList, None] = None,
        drop_remaining: bool = False,
    ) -> types.Logs:
        """Returns the loss value & metrics values for the model in test mode.
        Computation is done in batches.

        Arguments:
            inputs: Input data. It could be:

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
            labels: Target data. Like the input data `x`,
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
        [`Model.fit`][elegy.model.model.Model.fit].

        Returns:
            A dictionary for mapping the losses and metrics names to the values obtained.
        Raises:
            ValueError: in case of invalid arguments.
        """
        if batch_size is not None:
            batch_size = self.module.distributed_strategy.lift_batch_size(batch_size)

        if inputs is None:
            inputs = {}

        data_handler = data.DataHandler(
            x=inputs,
            y=labels,
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
                sigint_mode=SigIntMode.TEST,
                add_module=True,
                model=self,
                verbose=verbose,
                epochs=1,
                steps=data_handler.inferred_steps,
            )

        callbacks.on_test_begin()

        logs = {}
        for _, iterator in data_handler.enumerate_epochs():
            self.reset_step()
            with data_handler.catch_stop_iteration():
                for step in data_handler.steps():
                    callbacks.on_test_batch_begin(step)
                    batch = next(iterator)
                    # inputs, labels, _ = data.unpack_x_y_sample_weight(batch)

                    if drop_remaining and not data_utils.has_batch_size(
                        batch, data_handler.batch_size
                    ):
                        continue

                    if isinstance(batch, (tuple, list)) and len(batch) == 1:
                        batch = batch[0]

                    tmp_logs = self._make_test_step(batch, step)
                    tmp_logs.update({"size": data_handler.batch_size})
                    logs = tmp_logs
                    callbacks.on_test_batch_end(step, logs)

                    if self.stop_training:
                        break
            if self.stop_training:
                break

        callbacks.on_test_end()

        return logs

    def fit(
        self,
        inputs: tp.Optional[tp.Any] = None,
        labels: tp.Optional[tp.Any] = None,
        sample_weight: tp.Optional[tp.Any] = None,
        batch_size: tp.Optional[int] = 32,
        epochs: int = 1,
        verbose: int = 1,
        callbacks: tp.Union[tp.List[Callback], CallbackList, None] = None,
        validation_split: float = 0.0,
        validation_data: tp.Union[tp.Tuple, tp.Iterable, None] = None,
        shuffle: bool = True,
        initial_epoch: int = 0,
        steps_per_epoch: tp.Optional[int] = None,
        validation_steps: tp.Optional[int] = None,
        validation_batch_size: tp.Optional[int] = None,
        validation_freq: int = 1,
        drop_remaining: bool = True,
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
            verbose: 0, 1, 2 or 3. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch 3 = table.
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

        if batch_size is not None:
            batch_size = self.module.distributed_strategy.lift_batch_size(batch_size)

        if inputs is None:
            inputs = dict()

        if validation_split:
            # Create the validation data using the training data. Only supported for
            # `Jax Numpy` and `NumPy` input.
            (inputs, labels,), validation_data = data.train_validation_split(
                (inputs, labels),
                validation_split=validation_split,
                shuffle=False,
            )

        self.stop_training = False
        data_handler = data.DataHandler(
            x=inputs,
            y=labels,
            sample_weight=sample_weight,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            initial_epoch=initial_epoch,
            epochs=epochs,
            shuffle=shuffle,
            # class_weight=class_weight,
        )
        # Container that configures and calls `tf.keras.Callback`s.
        if not isinstance(callbacks, CallbackList):
            callbacks = CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                sigint_mode=SigIntMode.TRAIN,
                add_module=True,
                model=self,
                verbose=verbose,
                epochs=epochs,
                steps=data_handler.inferred_steps,
            )

        callbacks.on_train_begin()
        # data_handler._initial_epoch = (  # pylint: disable=protected-access
        #     self._maybe_load_initial_epoch_from_ckpt(initial_epoch))
        epoch_logs = {}

        for epoch, iterator in data_handler.enumerate_epochs():
            self.reset_step()
            callbacks.on_epoch_begin(epoch)
            logs = {}
            with data_handler.catch_stop_iteration():
                for step in data_handler.steps():
                    # prev_model = self.copy()

                    callbacks.on_train_batch_begin(step)
                    batch = next(iterator)

                    if drop_remaining and not data_utils.has_batch_size(
                        batch, data_handler.batch_size
                    ):
                        continue

                    if isinstance(batch, (tuple, list)) and len(batch) == 1:
                        batch = batch[0]

                    tmp_logs = self._make_train_step(batch, step, epoch)
                    tmp_logs.update({"size": data_handler.batch_size})

                    logs = tmp_logs
                    callbacks.on_train_batch_end(step, logs)

                    if self.stop_training:
                        break

                    # utils._walk_treedef(
                    #     jax.tree_flatten(prev_model)[1],
                    #     jax.tree_flatten(self)[1],
                    # )

            epoch_logs = copy(logs)
            epoch_logs.update({"size": data_handler.batch_size})

            # Run validation.
            if (
                validation_data
                and self._should_eval(epoch, validation_freq)
                and not self.stop_training
            ):
                (
                    val_inputs,
                    val_lables,
                    val_sample_weights,
                ) = data.unpack_x_y_sample_weight(validation_data)
                # val_inputs, val_lables = validation_data
                try:
                    val_logs = self.evaluate(
                        inputs=val_inputs,
                        labels=val_lables,
                        sample_weight=val_sample_weights,
                        batch_size=validation_batch_size or batch_size,
                        steps=validation_steps,
                        callbacks=callbacks,
                        # return_dict=True,
                        drop_remaining=drop_remaining,
                    )

                    val_logs = {"val_" + name: val for name, val in val_logs.items()}
                    epoch_logs.update(val_logs)
                except (types.MissingMethod, types.MissingModule) as e:
                    pass

            callbacks.on_epoch_end(epoch, epoch_logs)

            if self.stop_training:
                break

        callbacks.on_train_end(epoch_logs)

        assert self.history is not None

        return self.history

    def summary(
        self,
        *args,
        depth: int = 2,
        return_repr: bool = False,
        **kwargs,
    ) -> tp.Optional[str]:
        """
        Prints a summary of the network. The representation is module dependent,
        if a library provides a representation, it will be used, else elegy will
        define its own.

        Arguments:
            x: A sample of inputs to the network.
            depth: The level number of nested level which will be showed.
                Information about summaries from modules deeper than `depth`
                will be aggregated together.
            return_repr: If True, the summary will be returned as a string and will not be printed.
            eval_shape: If True, jax.eval_shape is used to calculate all shapes, this avoids actually
                running the computation as only shapes are calculated (turn off if trying to debug).
        """
        # model = self.local()

        # assert model.module is not None

        # if not model.initialized:
        #     if inputs is tx.MISSING:
        #         raise ValueError(
        #             "`inputs` is required to print the summary of uninitialized Models"
        #         )

        #     model.init_on_batch(inputs)

        summary = self.module.tabulate(*args, summary_depth=depth, **kwargs)

        if return_repr:
            return summary
        else:
            print(summary)

    def _should_eval(self, epoch, validation_freq):
        epoch = epoch + 1  # one-index the user-facing epoch.
        if isinstance(validation_freq, int):
            return epoch % validation_freq == 0
        elif isinstance(validation_freq, list):
            return epoch in validation_freq
        else:
            raise ValueError("Expected `validation_freq` to be a list or int.")
