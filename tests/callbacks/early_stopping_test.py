import logging
from unittest import TestCase

import jax
import numpy as np
import optax

import elegy as eg

np.random.seed(42)


class EarlyStoppingTest(TestCase):
    def test_example(self):
        class MLP(eg.Module):
            @eg.compact
            def __call__(self, x):
                x = eg.Linear(10)(x)
                x = jax.lax.stop_gradient(x)
                return x

        callback = eg.callbacks.EarlyStopping(monitor="loss", patience=3)
        # This callback will stop the training when there is no improvement in
        # the for three consecutive epochs.
        model = eg.Model(
            module=MLP(),
            loss=eg.losses.MeanSquaredError(),
            optimizer=optax.rmsprop(0.01),
        )
        history = model.fit(
            inputs=np.ones((5, 20)),
            labels=np.zeros((5, 10)),
            epochs=10,
            batch_size=1,
            callbacks=[callback],
            verbose=0,
        )
        assert len(history.history["loss"]) == 4  # Only 4 epochs are run.
