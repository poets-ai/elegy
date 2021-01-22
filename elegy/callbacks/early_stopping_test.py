import logging
from unittest import TestCase


import numpy as np
import optax
import jax

import elegy

np.random.seed(42)


class EarlyStoppingTest(TestCase):
    def test_example(self):
        class MLP(elegy.Module):
            def call(self, x):
                x = elegy.nn.Linear(10)(x)
                x = jax.lax.stop_gradient(x)
                return x

        callback = elegy.callbacks.EarlyStopping(monitor="loss", patience=3)
        # This callback will stop the training when there is no improvement in
        # the for three consecutive epochs.
        model = elegy.Model(
            module=MLP(),
            loss=elegy.losses.MeanSquaredError(),
            optimizer=optax.rmsprop(0.01),
        )
        history = model.fit(
            x=np.ones((5, 20)),
            y=np.zeros((5, 10)),
            epochs=10,
            batch_size=1,
            callbacks=[callback],
            verbose=0,
        )
        assert len(history.history["loss"]) == 4  # Only 4 epochs are run.
