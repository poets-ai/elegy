import copy
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
from .data import DataHandler, unpack_x_y_sample_weight
from .metrics.metric_modes import get_mode_function


class Model:
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
        optimizer: optix.GradientTransformation = optix.adam(1e-3),
        run_eagerly: bool = False,
        params: tp.Optional[hk.Params] = None,
        state: tp.Optional[hk.State] = None,
        optimizer_state: tp.Optional[optix.OptState] = None,
        metrics_state: tp.Optional[hk.State] = None,
        initial_metrics_state: tp.Optional[hk.State] = None,
        seed: tp.Union[jnp.ndarray, int] = jax.random.PRNGKey(42),
    ):

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
        self._optimizer = optimizer
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

        if self._optimizer_state is None:
            self._optimizer_state = self._optimizer.init(self._params)

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
        # verbose=1,
        callbacks=None,
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
    ):
        # if validation_split:
        #     # Create the validation data using the training data. Only supported for
        #     # `Tensor` and `NumPy` input.
        #     (
        #         (x, y, sample_weight),
        #         validation_data,
        #     ) = data_adapter.train_validation_split(
        #         (x, y, sample_weight), validation_split=validation_split, shuffle=False
        #     )

        # # Container that configures and calls `tf.keras.Callback`s.
        # if not isinstance(callbacks, callbacks_module.CallbackList):
        #     callbacks = callbacks_module.CallbackList(
        #         callbacks,
        #         add_history=True,
        #         add_progbar=verbose != 0,
        #         model=self,
        #         verbose=verbose,
        #         epochs=epochs,
        #         steps=data_handler.inferred_steps,
        #     )

        # callbacks.on_train_begin()
        # data_handler._initial_epoch = (  # pylint: disable=protected-access
        #     self._maybe_load_initial_epoch_from_ckpt(initial_epoch))
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
        for epoch, iterator in data_handler.enumerate_epochs():
            self.reset_metrics()
            # callbacks.on_epoch_begin(epoch)
            logs = {}
            with data_handler.catch_stop_iteration():
                for step in data_handler.steps():
                    # callbacks.on_train_batch_begin(step)
                    batch = next(iterator)
                    sample_weight = batch[2] if len(batch) == 3 else None

                    tmp_logs = self.train_on_batch(
                        x=batch[0],
                        y=batch[1],
                        sample_weight=sample_weight,
                        class_weight=class_weight,
                        # seed=None,
                        # params=None,
                        # state=None,
                        # optimizer_state=None,
                        # metrics_state=None,
                        # initial_metrics_state=None,
                    )
                    # print(epoch, step, tmp_logs, batch[0].shape)

                    logs = tmp_logs
                    # callbacks.on_train_batch_end(step, logs)

            epoch_logs = copy.copy(logs)

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

            # callbacks.on_epoch_end(epoch, epoch_logs)
            print(epoch, epoch_logs)
            if self.stop_training:
                break

        # callbacks.on_train_end()
        history = None
        return history

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
        batch_size: tp.Optional[int] = None,
        sample_weight: tp.Optional[tp.Union[np.ndarray, jnp.ndarray]] = None,
        steps: tp.Optional[int] = None,
        callbacks=None,
    ):

        # # Container that configures and calls `tf.keras.Callback`s.
        # if not isinstance(callbacks, callbacks_module.CallbackList):
        #     callbacks = callbacks_module.CallbackList(
        #         callbacks,
        #         add_history=True,
        #         add_progbar=verbose != 0,
        #         model=self,
        #         verbose=verbose,
        #         epochs=epochs,
        #         steps=data_handler.inferred_steps,
        #     )

        # callbacks.on_test_begin()

        data_handler = DataHandler(
            x=x,
            y=y,
            sample_weight=sample_weight,
            batch_size=batch_size,
            steps_per_epoch=steps,
            initial_epoch=0,
            epochs=1,
            shuffle=False,
        )
        logs = {}
        for _, iterator in data_handler.enumerate_epochs():
            self.reset_metrics()
            with data_handler.catch_stop_iteration():
                for step in data_handler.steps():
                    # callbacks.on_test_batch_begin(step)
                    batch = next(iterator)
                    sample_weight = batch[2] if len(batch) == 3 else None

                    tmp_logs = self.test_on_batch(
                        x=batch[0],
                        y=batch[1],
                        sample_weight=sample_weight,
                        # seed=None,
                        # params=None,
                        # state=None,
                        # optimizer_state=None,
                        # metrics_state=None,
                        # initial_metrics_state=None,
                    )

                    logs = tmp_logs
                    # callbacks.on_test_batch_end(step, logs)

        # callbacks.on_test_end(epoch, epoch_logs)

        return logs

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
