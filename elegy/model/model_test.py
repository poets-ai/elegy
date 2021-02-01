from hashlib import new
import unittest

import elegy
import jax
import jax.numpy as jnp
import numpy as np
import optax
import cloudpickle


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


class ModelBasicTest(unittest.TestCase):
    def test_predict(self):

        model = elegy.Model(module=elegy.nn.Linear(1))

        X = np.random.uniform(size=(5, 10))
        y = np.random.randint(10, size=(5, 1))

        y_pred = model.predict(x=X)

        assert y_pred.shape == (5, 1)

    def test_evaluate(self):
        def mse(y_true, y_pred):
            return jnp.mean((y_true - y_pred) ** 2)

        def mae(y_true, y_pred):
            return jnp.mean(jnp.abs(y_true - y_pred))

        model = elegy.Model(
            module=elegy.nn.Linear(1),
            loss=dict(a=mse),
            metrics=dict(b=mae),
            optimizer=optax.adamw(1e-3),
            run_eagerly=True,
        )

        X = np.random.uniform(size=(5, 10))
        y = np.random.uniform(size=(5, 1))

        logs = model.evaluate(x=X, y=y)

        assert "a/mse_loss" in logs
        assert "b/mae" in logs
        assert "loss" in logs

    def test_metrics(self):
        class M(elegy.Module):
            def call(self, x):
                n = self.add_parameter("n", lambda: 0, trainable=False)
                self.update_parameter("n", n + 1)
                return x

        metrics = elegy.model.model.Metrics(dict(a=dict(b=[M(), M()], c=M())))

        rng = elegy.RNGSeq(42)
        x = np.random.uniform(size=(5, 7, 7))

        with elegy.hooks.context(metrics=True):
            elegy.hooks.add_metric("d", 10)
            aux_metrics = elegy.hooks.get_metrics()
            logs, states = metrics.init(aux_metrics, rng)(x, training=True)

        with elegy.hooks.context(metrics=True):
            elegy.hooks.add_metric("d", 10)
            aux_metrics = elegy.hooks.get_metrics()
            logs, states = metrics.apply(aux_metrics, rng, states)(x, training=True)

        assert len(metrics.metrics) == 3
        assert "a/b/m" in metrics.metrics
        assert "a/b/m_1" in metrics.metrics
        assert "a/c/m" in metrics.metrics

        assert len(logs) == 4
        assert "a/b/m" in logs
        assert "a/b/m_1" in logs
        assert "a/c/m" in logs
        assert "d" in logs

        assert len(states) == 3
        assert "a/b/m" in states
        assert "a/b/m_1" in states
        assert "a/c/m" in states

    def test_losses(self):
        def loss_fn():
            return 3.0

        losses = elegy.model.model.Losses(dict(a=dict(b=[loss_fn, loss_fn], c=loss_fn)))

        rng = elegy.RNGSeq(42)
        hooks_losses = dict(x=0.3, y=4.5)

        with elegy.hooks.context(losses=True):
            elegy.hooks.add_loss("d", 1.0)
            aux_losses = elegy.hooks.get_losses()
            logs, logs, states = losses.init(aux_losses, rng)()

        with elegy.hooks.context(losses=True):
            elegy.hooks.add_loss("d", 1.0)
            aux_losses = elegy.hooks.get_losses()
            loss, logs, states = losses.apply(aux_losses, states)()

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

        newmodel.states = model.states
        newmodel.initial_states = model.initial_states

        y1 = newmodel.predict(X)
        assert np.all(y0 == y1)
