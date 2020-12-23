import unittest

import elegy
import jax
import jax.numpy as jnp
import numpy as np
import optax


class MLP(elegy.Module):
    """Standard LeNet-300-100 MLP network."""

    def __init__(self, n1: int = 3, n2: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.n1 = n1
        self.n2 = n2

    def call(self, image: jnp.ndarray, training: bool):
        x = image.astype(jnp.float32) / 255.0

        x = elegy.nn.Flatten()(x)
        x = elegy.nn.Linear(self.n1)(x)
        x = jax.nn.relu(x)

        x = elegy.nn.Linear(self.n2)(x)
        x = jax.nn.relu(x)
        x = elegy.nn.Linear(10)(x)

        return x


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
