import logging
from unittest import TestCase


import numpy as np
from jax.experimental import optix

import elegy

np.random.seed(42)


class ArrayDataAdapterTest(TestCase):
    def test_example(self):
        class MLP(elegy.Module):
            def call(self, input):
                mlp = elegy.nn.Sequential(lambda: [elegy.nn.Linear(10)])
                return mlp(input)

        callback = elegy.callbacks.EarlyStopping(monitor="loss", patience=3)
        # This callback will stop the training when there is no improvement in
        # the for three consecutive epochs.
        model = elegy.Model(
            module=MLP(),
            loss=elegy.losses.MeanSquaredError(),
            optimizer=optix.rmsprop(0.01),
        )
        history = model.fit(
            np.arange(100).reshape(5, 20).astype(np.float32),
            np.zeros(5),
            epochs=10,
            batch_size=1,
            callbacks=[callback],
            verbose=0,
        )
        assert len(history.history["loss"]) == 7  # Only 7 epochs are run.
