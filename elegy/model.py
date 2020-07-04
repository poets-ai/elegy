from functools import partial
import typing as tp
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import optix

from . import utils
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
    _run_eagerly: bool

    def __init__(
        self,
        model_fn: tp.Optional[tp.Callable],
        loss: tp.Callable[..., jnp.ndarray],
        metrics: tp.Optional[tp.Callable] = None,
        metrics_mode: tp.Union[str, tp.Callable] = "match_outputs_and_labels",
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
            if isinstance(metrics_mode, tp.Callable):
                metrics = metrics_mode(metrics)
            else:
                metrics = get_mode_function(metrics_mode)(metrics)

        self._model_fn = utils.DIFunction.create(model_fn)
        self._model_transform = hk.transform_with_state(self._model_fn)
        self._loss_fn = utils.DIFunction.create(loss)
        self._metrics_transform = (
            hk.transform_with_state(
                utils.DIFunction.create(metrics, rename={"__params": "params"})
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
        self._run_eagerly = run_eagerly

    def __call__(self, *args, **kwargs):
        return self._model_fn(*args, **kwargs)

    def _maybe_initialize(
        self,
        x: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple],
        y: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple, None],
        sample_weight: tp.Optional[jnp.ndarray] = None,
        class_weight: tp.Optional[jnp.ndarray] = None,
        seed: tp.Union[jnp.ndarray, int, None] = None,
        params: tp.Optional[hk.Params] = None,
        state: tp.Optional[hk.State] = None,
        optimizer_state: tp.Optional[optix.OptState] = None,
        metrics_state: tp.Optional[hk.State] = None,
        initial_metrics_state: tp.Optional[hk.State] = None,
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
            x_args, x_kwargs = self._get_input_args(x, y)

            self._params, self._state = self._model_transform.init(
                next(self._rngs), *x_args, **x_kwargs
            )

        if self._optimizer_state is None:
            self._optimizer_state = self._optimizer.init(self._params)

        if self._metrics_transform is not None and self._metrics_state is None:
            x_args, x_kwargs = self._get_input_args(x, y)

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

    def train_on_batch(
        self,
        x: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple],
        y: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple, None] = None,
        sample_weight: tp.Optional[jnp.ndarray] = None,
        class_weight: tp.Optional[jnp.ndarray] = None,
        reset_metrics: bool = False,
        seed: tp.Union[jnp.ndarray, int, None] = None,
        params: tp.Optional[hk.Params] = None,
        state: tp.Optional[hk.State] = None,
        optimizer_state: tp.Optional[optix.OptState] = None,
        metrics_state: tp.Optional[hk.State] = None,
        initial_metrics_state: tp.Optional[hk.State] = None,
    ) -> tp.Dict[str, jnp.ndarray]:
        def block():
            return self._train_on_batch(
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

        if self._run_eagerly:
            with jax.disable_jit():
                return block()
        else:
            return block()

    def reset_metrics(self, hard: bool = False):

        if hard:
            self._metrics_state = None
            self._initial_metrics_state = None
        else:
            self._metrics_state = self._initial_metrics_state

    def _train_on_batch(
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

        (
            logs,
            self._params,
            self._state,
            self._optimizer_state,
            self._metrics_state,
        ) = self._update(
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

    @partial(jax.jit, static_argnums=(0,))
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
        (loss, (y_pred, state)), grads = jax.value_and_grad(self._loss, has_aux=True)(
            params,
            state=state,
            net_rng=net_rng,
            x=x,
            y=y,
            sample_weight=sample_weight,
            class_weight=class_weight,
        )

        updates, optimizer_state = self._optimizer.update(grads, optimizer_state)
        params = optix.apply_updates(params, updates)

        logs = dict(loss=loss)

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

    def _loss(
        self,
        params: hk.Params,
        state: hk.State,
        net_rng: jnp.ndarray,
        x: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple],
        y: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple, None],
        sample_weight: tp.Optional[jnp.ndarray],
        class_weight: tp.Optional[jnp.ndarray],
    ):

        x_args, x_kwargs = self._get_input_args(x, y)
        y_pred, state = self._model_transform.apply(
            # required by apply
            params,
            state,
            net_rng,
            # new inputs + dependency injection
            *x_args,
            **x_kwargs,
        )

        loss = self._loss_fn(
            # required by loss API
            y,
            y_pred,
            # dependency injection
            x=x,
            sample_weight=sample_weight,
            class_weight=class_weight,
            params=params,
        )

        return loss, (y_pred, state)

    def _get_input_args(
        self,
        x: tp.Union[np.ndarray, jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple],
        y: tp.Any,
    ) -> tp.Tuple[tp.Tuple, tp.Mapping[str, tp.Any]]:

        if isinstance(x, tp.Tuple):
            args = x
            kwargs = {}
        elif isinstance(x, tp.Mapping):
            args = ()
            kwargs = x
        else:
            args = (x,)
            kwargs = {}

        apply_kwargs = dict(y=y)
        apply_kwargs.update(kwargs)

        return args, kwargs

