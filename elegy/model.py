import typing as tp
import haiku as hk
import jax
from jax.experimental.optix import OptState
import jax.numpy as jnp
import numpy as np
from jax.experimental import optix

from . import utils
from . import dependency_injection

OptState = tp.Union[optix.OptState]


class Model:
    net_fn: dependency_injection.DIFunction
    net: hk.Transformed
    loss_fn: dependency_injection.DIFunction
    optimizer: optix.GradientTransformation
    rngs: hk.PRNGSequence
    params: tp.Optional[hk.Params]
    optimizer_state: tp.Union[OptState, None]

    def __init__(
        self,
        loss: tp.Callable[..., jnp.ndarray],
        net_fn: tp.Optional[tp.Callable] = None,
        optimizer: optix.GradientTransformation = optix.adam(1e-3),
        seed: tp.Union[jax.random.PRNGKey, int] = jax.random.PRNGKey(42),
        use_jit: bool = False,
    ):

        if hasattr(self, "call"):
            net_fn = getattr(self, "call")

        if net_fn is None:
            raise ValueError("Must define either self.call or net_fn")

        self.net_fn = dependency_injection.DIFunction.create(net_fn)
        self.net = hk.transform(self.net_fn, apply_rng=True)
        self.loss_op = dependency_injection.DIFunction.create(loss)
        self.loss_fn = self.loss
        self.optimizer = optimizer
        self.rngs = hk.PRNGSequence(seed)
        self.params = None
        self.optimizer_state = None
        self.update_fn = jax.jit(self.update) if use_jit else self.update

    def __call__(self, *args, **kwargs):
        return self.net_fn(*args, **kwargs)

    def train_on_batch(
        self,
        x: jnp.ndarray,
        y: tp.Optional[jnp.ndarray] = None,
        seed: tp.Union[jnp.ndarray, int, None] = None,
        params: tp.Optional[hk.Params] = None,
    ) -> tp.Tuple[jnp.ndarray, hk.Params, OptState]:

        if seed is not None:
            self.rngs = hk.PRNGSequence(key_or_seed=seed)

        if params is not None:
            self.params = params

        if self.params is None:
            self.params = self.net.init(next(self.rngs), x=x, y=y, params=params)

        if self.optimizer_state is None:
            self.optimizer_state = self.optimizer.init(self.params)

        loss, self.params, self.optimizer_state = self.update_fn(
            self.params, self.optimizer_state, x, y
        )

        return loss, self.params, self.optimizer_state

    def loss(self, params: hk.Params, x: jnp.ndarray, y: tp.Optional[jnp.ndarray]):
        y_pred = self.net.apply(params, next(self.rngs), x=x, y=y)
        return self.loss_op(x=x, y_true=y, y_pred=y_pred, params=params)

    def update(
        self,
        params: hk.Params,
        optimizer_state: OptState,
        x: jnp.ndarray,
        y: tp.Optional[jnp.ndarray] = None,
    ) -> tp.Tuple[jnp.ndarray, hk.Params, OptState]:

        loss, grads = jax.value_and_grad(self.loss)(params, x, y)
        updates, optimizer_state = self.optimizer.update(grads, optimizer_state)
        params = optix.apply_updates(params, updates)

        return loss, params, optimizer_state

