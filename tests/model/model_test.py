import typing as tp
import unittest
from dataclasses import dataclass
from hashlib import new
from pathlib import Path
from tempfile import TemporaryDirectory

import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import sh
import tensorflow as tf

import elegy as eg


@dataclass
class MLP(eg.Module):
    dmid: int
    dout: int

    @eg.compact
    def __call__(self, x: jnp.ndarray):
        x = eg.Linear(self.dmid)(x)
        x = eg.BatchNorm()(x)
        x = jax.nn.relu(x)

        x = eg.Linear(self.dout)(x)
        return x


class ModelBasicTest(unittest.TestCase):
    def test_predict(self):

        model = eg.Model(module=eg.Linear(1))

        X = np.random.uniform(size=(5, 2))
        y = np.random.randint(10, size=(5, 1))

        y_pred = model.predict(X)

        assert y_pred.shape == (5, 1)

    def test_evaluate(self):
        class mse(eg.Loss):
            def call(self, target, preds):
                return jnp.mean((target - preds) ** 2)

        class mae(eg.Metric):
            value: eg.MetricState = eg.MetricState.node(
                default=jnp.array(0.0, jnp.float32)
            )

            def update(self, target, preds):
                return jnp.mean(jnp.abs(target - preds))

            def compute(self) -> tp.Any:
                return self.value

        model = eg.Model(
            module=eg.Linear(1),
            loss=dict(a=mse()),
            metrics=dict(b=mae()),
            optimizer=optax.adamw(1e-3),
            eager=True,
        )

        X = np.random.uniform(size=(5, 2))
        y = np.random.uniform(size=(5, 1))

        logs = model.evaluate(x=X, y=y)

        assert "a/mse_loss" in logs
        assert "b/mae" in logs
        assert "loss" in logs


class ModelTest(unittest.TestCase):
    def test_evaluate(self):

        model = eg.Model(
            module=MLP(dmid=3, dout=4),
            loss=[
                eg.losses.SparseCategoricalCrossentropy(from_logits=True),
                eg.regularizers.L2(l=1e-4),
            ],
            metrics=eg.metrics.Accuracy(),
            optimizer=optax.adamw(1e-3),
            eager=True,
        )

        X = np.random.uniform(size=(5, 2))
        y = np.random.randint(4, size=(5,))

        history = model.fit(
            inputs=X,
            labels=y,
            epochs=1,
            steps_per_epoch=1,
            batch_size=5,
            validation_data=(X, y),
            shuffle=True,
            verbose=1,
        )

        logs = model.evaluate(X, y)

        eval_acc = logs["accuracy"]
        predict_acc = (model.predict(X).argmax(-1) == y).mean()

        assert eval_acc == predict_acc

    def test_cloudpickle(self):
        model = eg.Model(
            module=MLP(dmid=3, dout=1),
            loss=[
                eg.losses.SparseCategoricalCrossentropy(from_logits=True),
                eg.regularizers.L2(1e-4),
            ],
            metrics=eg.metrics.Accuracy(),
            optimizer=optax.adamw(1e-3),
            eager=True,
        )

        X = np.random.uniform(size=(5, 2))
        y = np.random.randint(10, size=(5,))

        y0 = model.predict(X)

        model_pkl = cloudpickle.dumps(model)
        newmodel = cloudpickle.loads(model_pkl)

        y1 = newmodel.predict(X)
        assert np.all(y0 == y1)

    def test_saved_model(self):

        with TemporaryDirectory() as model_dir:

            model = eg.Model(module=eg.Linear(4))

            x = np.random.uniform(size=(5, 6))

            model.saved_model(x, model_dir, batch_size=[1, 2, 4, 8])

            output = str(sh.ls(model_dir))

            assert "saved_model.pb" in output
            assert "variables" in output

            saved_model = tf.saved_model.load(model_dir)

            saved_model

    def test_saved_model_poly(self):

        with TemporaryDirectory() as model_dir:

            model = eg.Model(module=eg.Linear(4))

            x = np.random.uniform(size=(5, 6)).astype(np.float32)

            model.saved_model(x, model_dir, batch_size=None)

            output = str(sh.ls(model_dir))

            assert "saved_model.pb" in output
            assert "variables" in output

            saved_model = tf.saved_model.load(model_dir)

            # change batch
            x = np.random.uniform(size=(3, 6)).astype(np.float32)
            y = saved_model(x)

            assert y.shape == (3, 4)


if __name__ == "__main__":
    ModelTest().test_mnist()
