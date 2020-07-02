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
    optimizer: optix.GradientTransformation
    rngs: hk.PRNGSequence
    params: tp.Optional[hk.Params]
    stage: tp.Optional[hk.State]
    optimizer_state: tp.Union[optix.OptState, None]

    def __init__(
        self,
        loss: tp.Callable[..., jnp.ndarray],
        net_fn: tp.Optional[tp.Callable] = None,
        optimizer: optix.GradientTransformation = optix.adam(1e-3),
        seed: tp.Union[jnp.ndarray, int] = jax.random.PRNGKey(42),
        run_eagerly: bool = False,
    ):

        if hasattr(self, "call"):
            net_fn = getattr(self, "call")

        if net_fn is None:
            raise ValueError("Must define either self.call or net_fn")

        self.net_fn = dependency_injection.DIFunction.create(net_fn)
        self.net = hk.transform_with_state(self.net_fn)
        self.loss_op = dependency_injection.DIFunction.create(loss)
        self.loss_fn = self.loss
        self.optimizer = optimizer
        self.rngs = hk.PRNGSequence(seed)
        self.params = None
        self.optimizer_state = None
        self.update_fn = self.update if run_eagerly else jax.jit(self.update)

    def __call__(self, *args, **kwargs):
        return self.net_fn(*args, **kwargs)

    def train_on_batch(
        self,
        x: jnp.ndarray,
        y: tp.Optional[jnp.ndarray] = None,
        seed: tp.Union[jnp.ndarray, int, None] = None,
        params: tp.Optional[hk.Params] = None,
    ) -> tp.Tuple[jnp.ndarray, hk.Params, hk.State, optix.OptState]:

        if seed is not None:
            self.rngs = hk.PRNGSequence(key_or_seed=seed)

        if params is not None:
            self.params = params

        if self.params is None:
            self.params, self.state = self.net.init(
                next(self.rngs), x, y=y, params=params
            )

        if self.optimizer_state is None:
            self.optimizer_state = self.optimizer.init(self.params)

        loss, self.params, self.state, self.optimizer_state = self.update_fn(
            self.params, self.state, self.optimizer_state, x, y
        )

        return loss, self.params, self.state, self.optimizer_state

    def loss(
        self,
        params: hk.Params,
        state: hk.State,
        x: jnp.ndarray,
        y: tp.Optional[jnp.ndarray],
    ):
        y_pred, state = self.net.apply(params, state, next(self.rngs), x=x, y=y)
        loss = self.loss_op(x=x, y_true=y, y_pred=y_pred, params=params)

        return loss, state

    def update(
        self,
        params: hk.Params,
        state: hk.State,
        optimizer_state: optix.OptState,
        x: jnp.ndarray,
        y: tp.Optional[jnp.ndarray] = None,
    ) -> tp.Tuple[jnp.ndarray, hk.Params, hk.State, optix.OptState]:

        (loss, state), grads = jax.value_and_grad(self.loss, has_aux=True)(
            params, state, x, y
        )
        updates, optimizer_state = self.optimizer.update(grads, optimizer_state)
        params = optix.apply_updates(params, updates)

        return loss, params, state, optimizer_state

