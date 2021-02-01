import jax
import elegy
import unittest
import numpy as np
import jax.numpy as jnp

import optax


class MLP(elegy.Module):
    """Standard LeNet-300-100 MLP network."""

    n1: int
    n2: int

    def __init__(self, n1: int = 3, n2: int = 4):
        super().__init__()
        self.n1 = n1
        self.n2 = n2

    def call(self, image: jnp.ndarray, training: bool):
        x = image.astype(jnp.float32) / 255.0

        x = jnp.reshape(x, [x.shape[0], -1])
        x = elegy.nn.Linear(self.n1)(x)
        x = elegy.nn.BatchNormalization()(x)
        x = jax.nn.relu(x)

        x = elegy.nn.Linear(self.n2)(x)
        x = jax.nn.relu(x)
        x = elegy.nn.Linear(10)(x)

        return x


class OptimizerTest(unittest.TestCase):
    def test_optimizer(self):
        optax_op = optax.adam(1e-3)
        lr_schedule = lambda step, epoch: step / 3

        optimizer = elegy.Optimizer(optax_op, lr_schedule=lr_schedule)

        params = np.random.uniform((3, 4))
        grads = np.random.uniform((3, 4))
        rng = elegy.RNGSeq(42)

        optimizer_states = optimizer.init(rng, params)
        assert jnp.allclose(optimizer.current_lr(optimizer_states), 0 / 3)

        params, optimizer_states = optimizer.apply(params, grads, optimizer_states, rng)
        assert jnp.allclose(optimizer.current_lr(optimizer_states), 1 / 3)

        params, optimizer_states = optimizer.apply(params, grads, optimizer_states, rng)
        assert jnp.allclose(optimizer.current_lr(optimizer_states), 2 / 3)

        params, optimizer_states = optimizer.apply(params, grads, optimizer_states, rng)
        assert jnp.allclose(optimizer.current_lr(optimizer_states), 3 / 3)

    def test_optimizer_epoch(self):
        optax_op = optax.adam(1e-3)
        lr_schedule = lambda step, epoch: epoch

        optimizer = elegy.Optimizer(
            optax_op, lr_schedule=lr_schedule, steps_per_epoch=2
        )

        params = np.random.uniform((3, 4))
        grads = np.random.uniform((3, 4))
        rng = elegy.RNGSeq(42)

        optimizer_states = optimizer.init(
            rng=rng,
            net_params=params,
        )

        assert jnp.allclose(optimizer.current_lr(optimizer_states), 0)
        params, optimizer_states = optimizer.apply(params, grads, optimizer_states, rng)

        assert jnp.allclose(optimizer.current_lr(optimizer_states), 0)
        params, optimizer_states = optimizer.apply(params, grads, optimizer_states, rng)

        assert jnp.allclose(optimizer.current_lr(optimizer_states), 1)
        params, optimizer_states = optimizer.apply(params, grads, optimizer_states, rng)

        assert jnp.allclose(optimizer.current_lr(optimizer_states), 1)
        params, optimizer_states = optimizer.apply(params, grads, optimizer_states, rng)

    def test_optimizer_chain(self):

        optimizer = elegy.Optimizer(
            optax.sgd(0.1),
            optax.clip(0.5),
        )

        params = np.zeros(shape=(3, 4))
        grads = np.ones(shape=(3, 4)) * 100_000
        rng = elegy.RNGSeq(42)

        optimizer_states = optimizer.init(
            rng=rng,
            net_params=params,
        )

        params, optimizer_states = optimizer.apply(params, grads, optimizer_states, rng)

        assert np.all(-0.5 <= params) and np.all(params <= 0.5)

    def test_lr_logging(self):
        model = elegy.Model(
            module=MLP(n1=3, n2=1),
            loss=elegy.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=elegy.metrics.SparseCategoricalAccuracy(),
            optimizer=elegy.Optimizer(
                optax.adamw(1.0, b1=0.95),
                lr_schedule=lambda step, epoch: jnp.array(1e-3),
            ),
            run_eagerly=True,
        )

        X = np.random.uniform(size=(5, 7, 7))
        y = np.random.randint(10, size=(5,))

        history = model.fit(
            x=X,
            y=y,
            epochs=1,
            steps_per_epoch=1,
            batch_size=5,
            validation_data=(X, y),
            shuffle=True,
            verbose=0,
        )

        assert "lr" in history.history
        assert np.allclose(history.history["lr"], 1e-3)
