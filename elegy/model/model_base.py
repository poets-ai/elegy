# Implementation based on tf.keras.engine.training.py
# https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/engine/training.py

import pickle
import typing as tp
from copy import copy
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from elegy import types
from elegy.callbacks import Callback, CallbackList, History
from elegy.data import (
    DataHandler,
    map_append,
    map_structure,
    train_validation_split,
    unpack_x_y_sample_weight,
)
from elegy.model.model_core import ModelCore, PredStep, TestStep
from tabulate import tabulate

# from elegy.module import Module


__all__ = ["Model", "load"]


class ModelBase(ModelCore):
    """
    `Model` is tasked with performing training, evaluation, and inference for a given
    `elegy.Module` or `haiku.Module`.

    To create a `Model` you first have to define its architecture in a `Module`:

    ```python
    >>> import elegy, jax
    >>> import jax.numpy as jnp

    >>> class MLP(elegy.Module):
    ...     def call(self, x: jnp.ndarray) -> jnp.ndarray:
    ...         x = elegy.nn.Flatten()(x)
    ...         x = elegy.nn.Linear(5)(x)
    ...         x = jax.nn.relu(x)
    ...         x = elegy.nn.Linear(2)(x)
    ...         return x

    >>> mlp = MLP()
    >>> x = jnp.ones(shape=[10, 2])

    >>> y_pred, collections = mlp.init(rng=elegy.RNGSeq(42))(x)
    >>> y_pred.shape
    (10, 2)

    ```


    You can pass use `Module` with the Model API:
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

    Model supports defining + monitoring custom learning rate schedules by passing an instance of `elegy.Optimizer` instead of
    an `optax` object:

    ```python
    model = elegy.Model(
        module=MLP(n1=3, n2=1),
        loss=elegy.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=elegy.metrics.SparseCategoricalAccuracy(),
        optimizer=elegy.Optimizer(
            optax.adam(1.0), # <---- important to set this to 1.0
            lr_schedule=lambda step, epoch: 1 / (epoch * 100 + step),
            steps_per_epoch=1000,
        ),
        run_eagerly=True,
    )

    history = model.fit(
        ...
    )

    assert "lr" in history.history
    ```
    Notice how we set the learning rate parameter of the `adam` optimizer to `1.0`, this is necessary if you want the logged `lr`
    be closer to the "actual" learning rate because we implement this feature by chaining an additional `optax.scale_by_schedule`
    at the end.
    """

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
        "states",
    ]

    def pred_step(self, *args, **kwargs):
        raise types.MissingMethod()

    def test_step(self, *args, **kwargs):
        raise types.MissingMethod()

    def train_step(self, *args, **kwargs):
        raise types.MissingMethod()

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
                try:
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
                except (types.MissingMethod, types.MissingModule) as e:
                    pass

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
    ) -> types.Logs:
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
        [`Model.fit`][elegy.model.model.Model.fit].

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


def load(path: tp.Union[str, Path]) -> ModelBase:
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
        include_optimizer: If True, loads optimizer's state if available.

    Raises:
        OSError: in case the model was not found or could not be
            loaded from disk successfully.
    """
    if isinstance(path, str):
        path = Path(path)

    try:
        model_bytes = (path / "model.pkl").read_bytes()
    except BaseException as e:
        raise OSError(f"Could not load the model. Got exception: {e}")

    model: ModelBase = pickle.loads(model_bytes)
    model.load(path)

    return model
