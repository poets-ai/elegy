from functools import partial
import typing as tp
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import optix

from . import utils
from . import dependency_injection


class Model:
    net_fn: tp.Callable
    net: hk.TransformedWithState
    loss_fn: tp.Callable
    metrics: tp.Optional[hk.TransformedWithState]
    optimizer: optix.GradientTransformation
    rngs: hk.PRNGSequence
    params: tp.Optional[hk.Params]
    state: tp.Optional[hk.State]
    optimizer_state: tp.Optional[optix.OptState]
    train_metrics_state: tp.Optional[hk.State]
    run_eagerly: bool

    def __init__(
        self,
        net_fn: tp.Optional[tp.Callable],
        loss: tp.Callable[..., jnp.ndarray],
        metrics: tp.Optional[tp.Callable] = None,
        optimizer: optix.GradientTransformation = optix.adam(1e-3),
        run_eagerly: bool = False,
        params: tp.Optional[hk.Params] = None,
        state: tp.Optional[hk.State] = None,
        optimizer_state: tp.Optional[optix.OptState] = None,
        train_metrics_state: tp.Optional[hk.State] = None,
        seed: tp.Union[jnp.ndarray, int] = jax.random.PRNGKey(42),
    ):

        if hasattr(self, "call"):
            net_fn = getattr(self, "call")

        if net_fn is None:
            raise ValueError("Must define either self.call or net_fn")

        self.net_fn = dependency_injection.DIFunction.create(net_fn)
        self.net = hk.transform_with_state(self.net_fn)
        self.loss_fn = dependency_injection.DIFunction.create(loss)
        self.metrics = (
            hk.transform_with_state(
                dependency_injection.DIFunction.create(
                    metrics, rename={"__params": "params"}
                )
            )
            if metrics
            else None
        )
        self.optimizer = optimizer
        self.rngs = hk.PRNGSequence(seed)
        self.params = params
        self.state = state
        self.optimizer_state = optimizer_state
        self.train_metrics_state = train_metrics_state
        self.run_eagerly = run_eagerly

    def __call__(self, *args, **kwargs):
        return self.net_fn(*args, **kwargs)

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
        train_metrics_state: tp.Optional[hk.State] = None,
    ) -> tp.Tuple[
        tp.Dict[str, jnp.ndarray], hk.Params, hk.State, optix.OptState, hk.State
    ]:
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
                train_metrics_state=train_metrics_state,
            )

        if self.run_eagerly:
            with jax.disable_jit():
                return block()
        else:
            return block()

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
        train_metrics_state: tp.Optional[hk.State],
    ) -> tp.Tuple[
        tp.Dict[str, jnp.ndarray], hk.Params, hk.State, optix.OptState, hk.State
    ]:

        if seed is not None:
            self.rngs = hk.PRNGSequence(key_or_seed=seed)

        if params is not None:
            self.params = params

        if state is not None:
            self.state = state

        if optimizer_state is not None:
            self.optimizer_state = optimizer_state

        if train_metrics_state is not None:
            self.train_metrics_state = train_metrics_state

        if self.params is None or self.state is None:
            x_args, x_kwargs = self.get_input_args(x, y)

            self.params, self.state = self.net.init(
                next(self.rngs), *x_args, **x_kwargs
            )

        if self.optimizer_state is None:
            self.optimizer_state = self.optimizer.init(self.params)

        if self.metrics is not None and self.train_metrics_state is None:
            x_args, x_kwargs = self.get_input_args(x, y)

            y_pred, state = self.net.apply(
                # required by apply
                self.params,
                self.state,
                next(self.rngs),
                # net inputs + DI
                *x_args,
                **x_kwargs,
            )
            _, self.train_metrics_state = self.metrics.init(
                # required by init
                next(self.rngs),
                # required by metric API
                y,
                y_pred,
                # DI
                x=x,
                sample_weight=sample_weight,
                class_weight=class_weight,
                __params=self.params,
            )

        (
            logs,
            self.params,
            self.state,
            self.optimizer_state,
            self.train_metrics_state,
        ) = self._update(
            x=x,
            y=y,
            sample_weight=sample_weight,
            class_weight=class_weight,
            params=self.params,
            state=self.state,
            optimizer_state=self.optimizer_state,
            train_metrics_state=self.train_metrics_state,
            net_rng=next(self.rngs),
            metrics_rng=next(self.rngs),
        )

        return (
            logs,
            self.params,
            self.state,
            self.optimizer_state,
            self.train_metrics_state,
        )

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
        train_metrics_state: tp.Optional[hk.State],
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
        # , grads = outputs

        updates, optimizer_state = self.optimizer.update(grads, optimizer_state)
        params = optix.apply_updates(params, updates)

        logs = dict(loss=loss)

        if self.metrics is not None:
            metrics, train_metrics_state = self.metrics.apply(
                # required by apply
                {},  # params
                train_metrics_state,  # state
                metrics_rng,  # rng
                # required by metric API
                y,
                y_pred,
                # DI
                x=x,
                sample_weight=sample_weight,
                class_weight=class_weight,
                __params=params,
            )

            logs.update(metrics)

        return logs, params, state, optimizer_state, train_metrics_state

    def _loss(self, params, state, net_rng, x, y, sample_weight, class_weight):

        x_args, x_kwargs = self.get_input_args(x, y)
        y_pred, state = self.net.apply(
            # required by apply
            params,
            state,
            net_rng,
            # new inputs + DI
            *x_args,
            **x_kwargs,
        )

        loss = self.loss_fn(
            # required by loss API
            y,
            y_pred,
            # DI
            x=x,
            sample_weight=sample_weight,
            class_weight=class_weight,
            params=params,
        )

        return loss, (y_pred, state)

    def get_input_args(
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

