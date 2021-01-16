import unittest

import elegy
import jax
import jax.numpy as jnp
import numpy as np
import optax
import cloudpickle
from flax import linen as nn


class MLP(nn.Module):
    """Standard LeNet-300-100 MLP network."""

    n1: int = 3
    n2: int = 4

    @classmethod
    def new(cls, n1: int = 3, n2: int = 4):
        return cls(n1=n1, n2=n2)

    @nn.compact
    def __call__(self, image: jnp.ndarray, training: bool):
        x = image.astype(jnp.float32) / 255.0

        x = jnp.reshape(x, [x.shape[0], -1])
        x = nn.Dense(self.n1)(x)
        x = nn.BatchNorm()(x)
        x = jax.nn.relu(x)

        x = nn.Dense(self.n2)(x)
        x = jax.nn.relu(x)
        x = nn.Dense(10)(x)

        return x


class ModelBasicTest(unittest.TestCase):
    def test_predict(self):

        model = elegy.Model(
            module=MLP.new(n1=3, n2=1),
            # loss=[
            #     elegy.losses.SparseCategoricalCrossentropy(from_logits=True),
            #     elegy.regularizers.GlobalL2(l=1e-4),
            # ],
            # metrics=elegy.metrics.SparseCategoricalAccuracy(),
            optimizer=optax.adamw(1e-3),
            run_eagerly=True,
        )

        X = np.random.uniform(size=(5, 7, 7))
        y = np.random.randint(10, size=(5,))

        y_pred = model.predict(x=X)

        assert y_pred.shape == (5, 10)

    def test_metrics(self):
        metrics = elegy.model.model.Metrics(dict(a=dict(b=[MLP(), MLP()], c=MLP())))

        rng = elegy.RNGSeq(42)
        x = np.random.uniform(size=(5, 7, 7))

        with elegy.hooks.hooks_context():
            elegy.add_metric("d", 10)
            logs, states = metrics.init(rng)(x, training=True)

        with elegy.hooks.hooks_context():
            elegy.add_metric("d", 10)
            logs, states = metrics.apply(states, rng)(x, training=True)

        assert len(metrics.metrics) == 3
        assert "a/b/mlp" in metrics.metrics
        assert "a/b/mlp_1" in metrics.metrics
        assert "a/c/mlp" in metrics.metrics

        assert len(logs) == 4
        assert "a/b/mlp" in logs
        assert "a/b/mlp_1" in logs
        assert "a/c/mlp" in logs
        assert "d" in logs

        assert len(states) == 3
        assert "a/b/mlp" in states
        assert "a/b/mlp_1" in states
        assert "a/c/mlp" in states

    def test_losses(self):
        def loss_fn():
            return 3.0

        losses = elegy.model.model.Losses(dict(a=dict(b=[loss_fn, loss_fn], c=loss_fn)))

        rng = elegy.RNGSeq(42)
        hooks_losses = dict(x=0.3, y=4.5)

        with elegy.hooks_context():
            elegy.add_loss("d", 1.0)
            logs, logs, states = losses.init(rng)()

        with elegy.hooks_context():
            elegy.add_loss("d", 1.0)
            loss, logs, states = losses.apply(states)()

        assert loss == 10

        assert len(losses.losses) == 3
        assert "a/b/loss_fn" in losses.losses
        assert "a/b/loss_fn_1" in losses.losses
        assert "a/c/loss_fn" in losses.losses

        assert len(logs) == 5
        assert "loss" in logs
        assert "a/b/loss_fn" in logs
        assert "a/b/loss_fn_1" in logs
        assert "a/c/loss_fn" in logs
        assert "d_loss" in logs


class ModelTest(unittest.TestCase):
    def test_evaluate(self):

        model = elegy.Model(
            module=MLP(n1=3, n2=1),
            loss=[
                elegy.losses.SparseCategoricalCrossentropy(from_logits=True),
                elegy.regularizers.GlobalL2(l=1e-4),
            ],
            metrics=elegy.metrics.SparseCategoricalAccuracy(),
            optimizer=optax.adamw(1e-3),
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
            verbose=1,
        )

        logs = model.evaluate(X, y)

        eval_acc = logs["sparse_categorical_accuracy"]
        predict_acc = (model.predict(X).argmax(-1) == y).mean()

        assert eval_acc == predict_acc

    def test_cloudpickle(self):
        model = elegy.Model(
            module=MLP(n1=3, n2=1),
            loss=[
                elegy.losses.SparseCategoricalCrossentropy(from_logits=True),
                elegy.regularizers.GlobalL2(l=1e-4),
            ],
            metrics=elegy.metrics.SparseCategoricalAccuracy(),
            optimizer=optax.adamw(1e-3),
            run_eagerly=True,
        )

        X = np.random.uniform(size=(5, 7, 7))

        y0 = model.predict(X)

        model_pkl = cloudpickle.dumps(model)
        newmodel = cloudpickle.loads(model_pkl)

        y1 = newmodel.predict(X)
        assert np.all(y0 == y1)

    def test_optimizer(self):
        optax_op = optax.adam(1e-3)
        lr_schedule = lambda step, epoch: step / 3

        optimizer = elegy.Optimizer(optax_op, lr_schedule=lr_schedule)

        params = np.random.uniform((3, 4))
        grads = np.random.uniform((3, 4))

        params = optimizer(params, grads)
        assert jnp.allclose(optimizer.get_effective_learning_rate(), 1 / 3)

        params = optimizer(params, grads)
        assert jnp.allclose(optimizer.get_effective_learning_rate(), 2 / 3)

        params = optimizer(params, grads)
        assert jnp.allclose(optimizer.get_effective_learning_rate(), 3 / 3)

    def test_optimizer_epoch(self):
        optax_op = optax.adam(1e-3)
        lr_schedule = lambda step, epoch: epoch

        optimizer = elegy.Optimizer(
            optax_op, lr_schedule=lr_schedule, steps_per_epoch=2
        )

        params = np.random.uniform((3, 4))
        grads = np.random.uniform((3, 4))

        params = optimizer.init(params, grads)
        assert jnp.allclose(optimizer.get_effective_learning_rate(), 0)

        params = optimizer(params, grads)
        assert jnp.allclose(optimizer.get_effective_learning_rate(), 0)

        params = optimizer(params, grads)
        assert jnp.allclose(optimizer.get_effective_learning_rate(), 1)

        params = optimizer(params, grads)
        assert jnp.allclose(optimizer.get_effective_learning_rate(), 1)

    def test_optimizer_chain(self):

        optimizer = elegy.Optimizer(
            optax.sgd(0.1),
            optax.clip(0.5),
        )

        params = np.zeros(shape=(3, 4))
        grads = np.ones(shape=(3, 4)) * 100_000

        params = optimizer(params, grads)

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
