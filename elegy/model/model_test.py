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


class LearningRatesMonitorTest(unittest.TestCase):
    def test_elegy_optax(self):
        sgd = elegy.optax.sgd(0.1, momentum=0.9)
        assert hasattr(sgd, "lr")
        assert sgd.lr == 0.1

        sched0 = elegy.optax.scale_by_schedule(
            optax.polynomial_schedule(0, 1, power=1, transition_steps=2)
        )
        sched1 = elegy.optax.scale_by_schedule(
            optax.polynomial_schedule(1, 0, power=1, transition_steps=5)
        )

        assert hasattr(sched0, "step_fns")
        assert len(sched0.step_fns) == 1 and len(sched1.step_fns) == 1

        chain = elegy.optax.chain(
            optax.additive_weight_decay(1e-4),
            sgd,
            sched0,
            sched1,
        )

        assert len(chain.step_fns) == 2
        assert chain.lr == sgd.lr

    def test_optimizer(self):
        chain = elegy.optax.chain(
            optax.additive_weight_decay(1e-4),
            elegy.optax.sgd(0.1, momentum=0.9),
            elegy.optax.scale_by_schedule(
                optax.polynomial_schedule(0, 1, power=1, transition_steps=2)
            ),
            elegy.optax.scale_by_schedule(
                optax.polynomial_schedule(1, 0, power=1, transition_steps=5)
            ),
        )

        optimizer = elegy.model.model_base.Optimizer(chain)
        parameter = 0
        gradient = 0

        assert optimizer.get_effective_learning_rate() == 0.0

        # lr increases
        optimizer(parameter, gradient)
        elr1 = optimizer.get_effective_learning_rate()
        assert elr1 > 0

        optimizer(parameter, gradient)
        elr2 = optimizer.get_effective_learning_rate()
        assert elr2 > elr1

        # lr decreases
        optimizer(parameter, gradient)
        elr3 = optimizer.get_effective_learning_rate()
        assert elr3 < elr2

    def test_lr_logging(self):
        model = elegy.Model(
            module=MLP(n1=3, n2=1),
            loss=elegy.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=elegy.metrics.SparseCategoricalAccuracy(),
            optimizer=elegy.optax.adamw(1e-3),
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

        print(history)
        print(elegy.optax.__dict__.keys())
        assert "lr" in history.history
        assert np.allclose(history.history["lr"], 1e-3)
