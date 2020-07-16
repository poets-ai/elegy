# Implementation based on tf.keras.engine.training.py
# https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/engine/training.py

import copy
from enum import Enum
import typing as tp
from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import optix

from elegy.losses import loss_modes
from elegy.metrics import metric_modes

from . import utils
from .data import (
    DataHandler,
    unpack_x_y_sample_weight,
    train_validation_split,
    map_structure,
    map_append,
)
from .metrics.metric_modes import get_mode_function
from .callbacks import CallbackList, Callback


class Mode(Enum):
    predict = 1
    test = 2
    train = 3


class Model(object):
    _model_fn: tp.Callable
    _model_transform: hk.TransformedWithState
    _loss_fn: tp.Callable
    _metrics: tp.Optional[hk.TransformedWithState]
    _optimizer: optix.GradientTransformation
    _rngs: hk.PRNGSequence
    _params: tp.Optional[hk.Params]
    _state: tp.Optional[hk.State]
    _optimizer_state: tp.Optional[optix.OptState]
    _metrics_state: tp.Optional[hk.State]
    _initial_metrics_state: tp.Optional[hk.State]
    run_eagerly: bool

    def __init__(
        self,
        model_fn: tp.Optional[tp.Callable],
        loss: tp.Callable,
        loss_mode: str = "match_outputs_and_labels",
        aux_losses: tp.Optional[
            tp.Callable[[], tp.Union[tp.List[tp.Callable], tp.Callable]]
        ] = None,
        metrics: tp.Optional[tp.Callable] = None,
        metrics_mode: str = "match_outputs_and_labels",
        optimizer: tp.Optional[optix.GradientTransformation] = None,
        run_eagerly: bool = False,
        params: tp.Optional[hk.Params] = None,
        state: tp.Optional[hk.State] = None,
        optimizer_state: tp.Optional[optix.OptState] = None,
        metrics_state: tp.Optional[hk.State] = None,
        initial_metrics_state: tp.Optional[hk.State] = None,
        seed: tp.Union[jnp.ndarray, int] = jax.random.PRNGKey(42),
    ):
        """[summary]

        Args:
            model_fn (tp.Optional[tp.Callable]): [description]
            loss (tp.Callable): [description]
            loss_mode (str, optional): [description]. Defaults to "match_outputs_and_labels".
            aux_losses (tp.Optional[ tp.Callable[[], tp.Union[tp.List[tp.Callable], tp.Callable]] ], optional): [description]. Defaults to None.
            metrics (tp.Optional[tp.Callable], optional): [description]. Defaults to None.
            metrics_mode (str, optional): [description]. Defaults to "match_outputs_and_labels".
            optimizer (optix.GradientTransformation, optional): [description]. Defaults to optix.adam(1e-3).
            run_eagerly (bool, optional): [description]. Defaults to False.
            params (tp.Optional[hk.Params], optional): [description]. Defaults to None.
            state (tp.Optional[hk.State], optional): [description]. Defaults to None.
            optimizer_state (tp.Optional[optix.OptState], optional): [description]. Defaults to None.
            metrics_state (tp.Optional[hk.State], optional): [description]. Defaults to None.
            initial_metrics_state (tp.Optional[hk.State], optional): [description]. Defaults to None.
            seed (tp.Union[jnp.ndarray, int], optional): [description]. Defaults to jax.random.PRNGKey(42).

        Raises:
            ValueError: [description]
        """

        if hasattr(self, "call"):
            model_fn = getattr(self, "call")

        if model_fn is None:
            raise ValueError("Must define either self.call or model_fn")

        if metrics is not None:
            metrics = metric_modes.get_mode_function(metrics_mode)(metrics)

        loss = loss_modes.get_mode_function(loss_mode)(loss)

        self._model_fn = utils.inject_dependencies(model_fn)
        self._model_transform = hk.transform_with_state(self._model_fn)
        self._loss_fn = utils.inject_dependencies(loss)
        self._aux_losses = (
            loss_modes.get_aux_losses_fn(aux_losses) if aux_losses is not None else None
        )
        self._metrics_transform = (
            hk.transform_with_state(
                utils.inject_dependencies(metrics, rename={"__params": "params"})
            )
            if metrics
            else None
        )
        self._optimizer = optimizer if optimizer is not None else optix.adam(1e-3)
        self._rngs = hk.PRNGSequence(seed)
        self._params = params
        self._state = state
        self._optimizer_state = optimizer_state
        self._metrics_state = metrics_state
        self._initial_metrics_state = initial_metrics_state
        self.run_eagerly = run_eagerly

    def __call__(self, *args, **kwargs):
        return self._model_fn(*args, **kwargs)

    def _maybe_initialize(
        self,
        x: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple],
        y: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple, None],
        sample_weight: tp.Optional[jnp.ndarray],
        class_weight: tp.Optional[jnp.ndarray],
        seed: tp.Union[jnp.ndarray, int, None],
        params: tp.Optional[hk.Params],
        state: tp.Optional[hk.State],
        optimizer_state: tp.Optional[optix.OptState],
        metrics_state: tp.Optional[hk.State],
        initial_metrics_state: tp.Optional[hk.State],
        mode: Mode,
    ):

        if seed is not None:
            self._rngs = hk.PRNGSequence(key_or_seed=seed)

        if params is not None:
            self._params = params

        if state is not None:
            self._state = state

        if optimizer_state is not None:
            self._optimizer_state = optimizer_state

        if metrics_state is not None:
            self._metrics_state = metrics_state

        if initial_metrics_state is not None:
            self._initial_metrics_state = initial_metrics_state

        if self._params is None or self._state is None:
            x_args, x_kwargs = utils.get_input_args(x, y, is_training=False)

            self._params, self._state = self._model_transform.init(
                next(self._rngs), *x_args, **x_kwargs
            )

        if mode == Mode.predict:
            return

        if self._metrics_transform is not None and self._metrics_state is None:
            x_args, x_kwargs = utils.get_input_args(x, y, is_training=False)

            y_pred, state = self._model_transform.apply(
                # required by apply
                self._params,
                self._state,
                next(self._rngs),
                # model_transform inputs + dependency injection
                *x_args,
                **x_kwargs,
            )
            _, self._metrics_state = self._metrics_transform.init(
                # required by init
                next(self._rngs),
                # required by metric API
                y_true=y,
                y_pred=y_pred,
                # dependency injection
                x=x,
                sample_weight=sample_weight,
                class_weight=class_weight,
                __params=self._params,  # renamed
            )

            self._initial_metrics_state = self._metrics_state

        if mode == Mode.test:
            return

        if self._optimizer_state is None:
            self._optimizer_state = self._optimizer.init(self._params)

    def reset_metrics(self, hard: bool = False):

        if hard:
            self._metrics_state = None
            self._initial_metrics_state = None
        else:
            self._metrics_state = self._initial_metrics_state

    def train_on_batch(
        self,
        x: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple],
        y: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple, None] = None,
        sample_weight: tp.Optional[jnp.ndarray] = None,
        class_weight: tp.Optional[jnp.ndarray] = None,
        seed: tp.Union[jnp.ndarray, int, None] = None,
        params: tp.Optional[hk.Params] = None,
        state: tp.Optional[hk.State] = None,
        optimizer_state: tp.Optional[optix.OptState] = None,
        metrics_state: tp.Optional[hk.State] = None,
        initial_metrics_state: tp.Optional[hk.State] = None,
    ) -> tp.Dict[str, jnp.ndarray]:

        self._maybe_initialize(
            x=x,
            y=y,
            sample_weight=sample_weight,
            class_weight=class_weight,
            seed=seed,
            params=params,
            state=state,
            optimizer_state=optimizer_state,
            metrics_state=metrics_state,
            initial_metrics_state=initial_metrics_state,
            mode=Mode.train,
        )

        update_fn = self._update if self.run_eagerly else self._update_jit

        (
            logs,
            self._params,
            self._state,
            self._optimizer_state,
            self._metrics_state,
        ) = update_fn(
            x=x,
            y=y,
            sample_weight=sample_weight,
            class_weight=class_weight,
            params=self._params,
            state=self._state,
            optimizer_state=self._optimizer_state,
            metrics_state=self._metrics_state,
            net_rng=next(self._rngs),
            metrics_rng=next(self._rngs),
        )

        return {key: np.asarray(value) for key, value in logs.items()}

    def _update(
        self,
        x: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple],
        y: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple, None],
        sample_weight: tp.Optional[jnp.ndarray],
        class_weight: tp.Optional[jnp.ndarray],
        params: hk.Params,
        state: hk.State,
        optimizer_state: optix.OptState,
        metrics_state: tp.Optional[hk.State],
        net_rng: jnp.ndarray,
        metrics_rng: jnp.ndarray,
    ) -> tp.Tuple[
        tp.Dict[str, jnp.ndarray],
        hk.Params,
        hk.State,
        optix.OptState,
        tp.Optional[hk.State],
    ]:
        (loss, (y_pred, state, logs)), grads = jax.value_and_grad(
            self._loss, has_aux=True
        )(
            params,
            state=state,
            net_rng=net_rng,
            x=x,
            y=y,
            sample_weight=sample_weight,
            class_weight=class_weight,
            is_training=True,
        )

        updates, optimizer_state = self._optimizer.update(grads, optimizer_state)
        params = optix.apply_updates(params, updates)

        if self._metrics_transform is not None:
            metrics, metrics_state = self._metrics_transform.apply(
                # required by apply
                {},  # params
                metrics_state,  # state
                metrics_rng,  # rng
                # required by metric API
                y_true=y,
                y_pred=y_pred,
                # dependency injection
                x=x,
                sample_weight=sample_weight,
                class_weight=class_weight,
                __params=params,  # renamed
            )
            logs.update(metrics)

        return logs, params, state, optimizer_state, metrics_state

    _update_jit = jax.jit(_update, static_argnums=(0,))

    def _loss(
        self,
        params: hk.Params,
        state: hk.State,
        net_rng: jnp.ndarray,
        x: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple],
        y: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple, None],
        sample_weight: tp.Optional[jnp.ndarray],
        class_weight: tp.Optional[jnp.ndarray],
        is_training: bool,
    ):

        y_pred, state = self._predict(
            x=x, params=params, state=state, net_rng=net_rng, is_training=is_training,
        )

        logs = self._loss_fn(
            # required by loss API
            y_true=y,
            y_pred=y_pred,
            # dependency injection
            x=x,
            sample_weight=sample_weight,
            class_weight=class_weight,
            params=params,
        )

        if not isinstance(logs, dict):
            logs = dict(loss=logs)

        aux_losses = (
            self._aux_losses(
                y_true=y,
                y_pred=y_pred,
                x=x,
                sample_weight=sample_weight,
                class_weight=class_weight,
                params=params,
            )
            if self._aux_losses is not None
            else None
        )

        if aux_losses:
            temp_logs = logs.copy()
            logs = aux_losses
            logs.update(temp_logs)

        # get total loss
        loss = logs["loss"] = sum(logs.values())

        return loss, (y_pred, state, logs)

    def fit(
        self,
        x: tp.Union[
            jnp.ndarray,
            np.ndarray,
            tp.Mapping[str, tp.Union[np.ndarray, jnp.ndarray]],
            tp.Tuple[tp.Union[np.ndarray, jnp.ndarray]],
            tp.Iterable,
        ],
        y: tp.Union[
            jnp.ndarray,
            np.ndarray,
            tp.Mapping[str, tp.Union[np.ndarray, jnp.ndarray]],
            tp.Tuple[tp.Union[np.ndarray, jnp.ndarray]],
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
        sample_weight: tp.Optional[tp.Union[np.ndarray, jnp.ndarray]] = None,
        initial_epoch: int = 0,
        steps_per_epoch: tp.Optional[int] = None,
        validation_steps: tp.Optional[int] = None,
        validation_batch_size: tp.Optional[int] = None,
        validation_freq: int = 1,
        seed: tp.Union[jnp.ndarray, int, None] = None,
        params: tp.Optional[hk.Params] = None,
        state: tp.Optional[hk.State] = None,
        optimizer_state: tp.Optional[optix.OptState] = None,
        metrics_state: tp.Optional[hk.State] = None,
        initial_metrics_state: tp.Optional[hk.State] = None,
    ):
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
            callbacks: List of [elegy.callbacks.Callback][] instances.
                List of callbacks to apply during training.
                See [elegy.callbacks][].
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
            A common pattern is to pass a tf.data.Dataset, generator, or
        elegy.utils.Sequence to the `x` argument of fit, which will in fact
        yield not only features (x) but optionally targets (y) and sample weights.
        Keras requires that the output of such iterator-likes be unambiguous. The
        iterator should return a tuple of length 1, 2, or 3, where the optional
        second and third elements will be used for y and sample_weight
        respectively. Any other type provided will be wrapped in a length one
        tuple, effectively treating everything as 'x'. When yielding dicts, they
        should still adhere to the top-level tuple structure.
        e.g. `({"x0": x0, "x1": x1}, y)`. Keras will not attempt to separate
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
        [elegy.model.Model.evaluate][] [elegy.model.Model][]
        Returns:
            A `History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).
        Raises:
            ValueError: In case of mismatch between the provided input data
                and what the model expects.
        """
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

            epoch_logs = copy.copy(logs)
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
            jnp.ndarray,
            np.ndarray,
            tp.Mapping[str, tp.Union[np.ndarray, jnp.ndarray]],
            tp.Tuple[tp.Union[np.ndarray, jnp.ndarray]],
            tp.Iterable,
        ],
        y: tp.Union[
            jnp.ndarray,
            np.ndarray,
            tp.Mapping[str, tp.Union[np.ndarray, jnp.ndarray]],
            tp.Tuple[tp.Union[np.ndarray, jnp.ndarray]],
            None,
        ] = None,
        verbose: int = 1,
        batch_size: tp.Optional[int] = None,
        sample_weight: tp.Optional[tp.Union[np.ndarray, jnp.ndarray]] = None,
        steps: tp.Optional[int] = None,
        callbacks: tp.Union[tp.List[Callback], CallbackList, None] = None,
    ):
        """Returns the loss value & metrics values for the model in test mode.
            Computation is done in batches.
            Arguments:
                x: Input data. It could be: - A Numpy array (or array-like), or a list
                of arrays (in case the model has multiple inputs). - A TensorFlow
                tensor, or a list of tensors (in case the model has multiple inputs).
                - A dict mapping input names to the corresponding array/tensors, if
                the model has named inputs. - A `tf.data` dataset. - A generator or
                `keras.utils.Sequence` instance. A more detailed description of
                unpacking behavior for iterator types (Dataset, generator, Sequence)
                is given in the `Unpacking behavior for iterator-like inputs` section
                of `Model.fit`.
                y: Target data. Like the input data `x`, it could be either Numpy
                array(s) or TensorFlow tensor(s). It should be consistent with `x`
                (you cannot have Numpy inputs and tensor targets, or inversely). If
                `x` is a dataset, generator or `keras.utils.Sequence` instance, `y`
                should not be specified (since targets will be obtained from the
                iterator/dataset).
                batch_size: Integer or `None`. Number of samples per gradient update. If
                unspecified, `batch_size` will default to 32. Do not specify the
                `batch_size` if your data is in the form of a dataset, generators,
                or `keras.utils.Sequence` instances (since they generate batches).
                verbose: 0 or 1. Verbosity mode. 0 = silent, 1 = progress bar.
                sample_weight: Optional Numpy array of weights for the test samples,
                used for weighting the loss function. You can either pass a flat (1D)
                Numpy array with the same length as the input samples
                    (1:1 mapping between weights and samples), or in the case of
                    temporal data, you can pass a 2D array with shape `(samples,
                    sequence_length)`, to apply a different weight to every timestep
                    of every sample. In this case you should make sure to specify
                    `sample_weight_mode="temporal"` in `compile()`. This argument is
                    not supported when `x` is a dataset, instead pass sample weights
                    as the third element of `x`.
                steps: Integer or `None`. Total number of steps (batches of samples)
                before declaring the evaluation round finished. Ignored with the
                default value of `None`. If x is a `tf.data` dataset and `steps` is
                None, 'evaluate' will run until the dataset is exhausted. This
                argument is not supported with array inputs.
                callbacks: List of `keras.callbacks.Callback` instances. List of
                callbacks to apply during evaluation. See
                [callbacks](/api_docs/python/tf/keras/callbacks).
                max_queue_size: Integer. Used for generator or `keras.utils.Sequence`
                input only. Maximum size for the generator queue. If unspecified,
                `max_queue_size` will default to 10.
                workers: Integer. Used for generator or `keras.utils.Sequence` input
                only. Maximum number of processes to spin up when using process-based
                threading. If unspecified, `workers` will default to 1. If 0, will
                execute the generator on the main thread.
                use_multiprocessing: Boolean. Used for generator or
                `keras.utils.Sequence` input only. If `True`, use process-based
                threading. If unspecified, `use_multiprocessing` will default to
                `False`. Note that because this implementation relies on
                multiprocessing, you should not pass non-picklable arguments to the
                generator as they can't be passed easily to children processes.
                return_dict: If `True`, loss and metric results are returned as a dict,
                with each key being the name of the metric. If `False`, they are
                returned as a list.
            See the discussion of `Unpacking behavior for iterator-like inputs` for
            `Model.fit`.
            Returns:
                Scalar test loss (if the model has a single output and no metrics)
                or list of scalars (if the model has multiple outputs
                and/or metrics). The attribute `model.metrics_names` will give you
                the display labels for the scalar outputs.
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
            is_training=False,
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
                        x=x_batch, y=y_batch, sample_weight=sample_weight,
                    )
                    tmp_logs.update({"size": data_handler.batch_size})
                    logs = tmp_logs
                    callbacks.on_test_batch_end(step, logs)

        callbacks.on_test_end()

        return logs

    def predict(
        self,
        x: tp.Union[
            jnp.ndarray,
            np.ndarray,
            tp.Mapping[str, tp.Union[np.ndarray, jnp.ndarray]],
            tp.Tuple[tp.Union[np.ndarray, jnp.ndarray]],
            tp.Iterable,
        ],
        verbose: int = 0,
        batch_size: tp.Optional[int] = None,
        steps: tp.Optional[int] = None,
        callbacks: tp.Union[tp.List[Callback], CallbackList, None] = None,
    ):
        """Generates output predictions for the input samples.
        Computation is done in batches. This method is designed for performance in
        large scale inputs. For small amount of inputs that fit in one batch,
        directly using `__call__` is recommended for faster execution, e.g.,
        `model(x)`, or `model(x, training=False)` if you have layers such as
        `tf.keras.layers.BatchNormalization` that behaves differently during
        inference.
        Arguments:
            x: Input samples. It could be:
            - A Numpy array (or array-like), or a list of arrays
                (in case the model has multiple inputs).
            - A TensorFlow tensor, or a list of tensors
                (in case the model has multiple inputs).
            - A `tf.data` dataset.
            - A generator or `keras.utils.Sequence` instance.
            A more detailed description of unpacking behavior for iterator types
            (Dataset, generator, Sequence) is given in the `Unpacking behavior
            for iterator-like inputs` section of `Model.fit`.
            batch_size: Integer or `None`.
                Number of samples per batch.
                If unspecified, `batch_size` will default to 32.
                Do not specify the `batch_size` if your data is in the
                form of dataset, generators, or `keras.utils.Sequence` instances
                (since they generate batches).
            verbose: Verbosity mode, 0 or 1.
            steps: Total number of steps (batches of samples)
                before declaring the prediction round finished.
                Ignored with the default value of `None`. If x is a `tf.data`
                dataset and `steps` is None, `predict` will
                run until the input dataset is exhausted.
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during prediction.
                See [callbacks](/api_docs/python/tf/keras/callbacks).
            max_queue_size: Integer. Used for generator or `keras.utils.Sequence`
                input only. Maximum size for the generator queue.
                If unspecified, `max_queue_size` will default to 10.
            workers: Integer. Used for generator or `keras.utils.Sequence` input
                only. Maximum number of processes to spin up when using
                process-based threading. If unspecified, `workers` will default
                to 1. If 0, will execute the generator on the main thread.
            use_multiprocessing: Boolean. Used for generator or
                `keras.utils.Sequence` input only. If `True`, use process-based
                threading. If unspecified, `use_multiprocessing` will default to
                `False`. Note that because this implementation relies on
                multiprocessing, you should not pass non-picklable arguments to
                the generator as they can't be passed easily to children processes.
        See the discussion of `Unpacking behavior for iterator-like inputs` for
        `Model.fit`. Note that Model.predict uses the same interpretation rules as
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

                        outputs = map_structure(map_append, outputs, batch_outputs,)

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
        seed: tp.Union[jnp.ndarray, int, None] = None,
        params: tp.Optional[hk.Params] = None,
        state: tp.Optional[hk.State] = None,
        metrics_state: tp.Optional[hk.State] = None,
        initial_metrics_state: tp.Optional[hk.State] = None,
    ) -> tp.Dict[str, jnp.ndarray]:

        self._maybe_initialize(
            x=x,
            y=y,
            sample_weight=sample_weight,
            class_weight=class_weight,
            seed=seed,
            params=params,
            state=state,
            optimizer_state=None,
            metrics_state=metrics_state,
            initial_metrics_state=initial_metrics_state,
            mode=Mode.test,
        )

        test_fn = self._test if self.run_eagerly else self._test_jit

        (logs, self._metrics_state,) = test_fn(
            x=x,
            y=y,
            sample_weight=sample_weight,
            class_weight=class_weight,
            params=self._params,
            state=self._state,
            optimizer_state=self._optimizer_state,
            metrics_state=self._metrics_state,
            net_rng=next(self._rngs),
            metrics_rng=next(self._rngs),
        )

        return {key: np.asarray(value) for key, value in logs.items()}

    def _test(
        self,
        x: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple],
        y: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple, None],
        sample_weight: tp.Optional[jnp.ndarray],
        class_weight: tp.Optional[jnp.ndarray],
        params: hk.Params,
        state: hk.State,
        optimizer_state: optix.OptState,
        metrics_state: tp.Optional[hk.State],
        net_rng: jnp.ndarray,
        metrics_rng: jnp.ndarray,
    ) -> tp.Tuple[
        tp.Dict[str, jnp.ndarray], tp.Optional[hk.State],
    ]:
        loss, (y_pred, _state, logs) = self._loss(
            params,
            state=state,
            net_rng=net_rng,
            x=x,
            y=y,
            sample_weight=sample_weight,
            class_weight=class_weight,
            is_training=False,
        )

        if self._metrics_transform is not None:
            metrics, metrics_state = self._metrics_transform.apply(
                # required by apply
                {},  # params
                metrics_state,  # state
                metrics_rng,  # rng
                # required by metric API
                y_true=y,
                y_pred=y_pred,
                # dependency injection
                x=x,
                sample_weight=sample_weight,
                class_weight=class_weight,
                __params=params,  # renamed
            )
            logs.update(metrics)

        return logs, metrics_state

    _test_jit = jax.jit(_test, static_argnums=(0,))
    # ----------------------------------------------------------------
    # predict
    # ----------------------------------------------------------------

    def predict_on_batch(
        self,
        x: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple],
        seed: tp.Union[jnp.ndarray, int, None] = None,
        params: tp.Optional[hk.Params] = None,
        state: tp.Optional[hk.State] = None,
    ) -> tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple]:

        self._maybe_initialize(
            x=x,
            y=None,
            sample_weight=None,
            class_weight=None,
            seed=seed,
            params=params,
            state=state,
            optimizer_state=None,
            metrics_state=None,
            initial_metrics_state=None,
            mode=Mode.predict,
        )

        predict_fn = self._predict if self.run_eagerly else self._predict_jit

        y_pred, _ = predict_fn(
            x=x,
            params=self._params,
            state=self._state,
            net_rng=next(self._rngs),
            is_training=False,
        )

        return y_pred

    def _predict(
        self,
        x: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple],
        params: hk.Params,
        state: hk.State,
        net_rng: jnp.ndarray,
        is_training: bool,
    ) -> tp.Tuple[
        tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple], hk.State,
    ]:
        x_args, x_kwargs = utils.get_input_args(x, None, is_training=is_training)
        y_pred, state = self._model_transform.apply(
            # required by apply
            params,
            state,
            net_rng,
            # new inputs + dependency injection
            *x_args,
            **x_kwargs,
        )

        return y_pred, state

    _predict_jit = jax.jit(_predict, static_argnums=(0,))

