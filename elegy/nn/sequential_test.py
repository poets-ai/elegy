import jax.numpy as jnp

from unittest import TestCase
import jax

import elegy


class SequentialTest(TestCase):
    #
    def test_connects(self):
        elegy.nn.Sequential(
            lambda: [
                elegy.nn.Flatten(),
                elegy.nn.Linear(5),
                jax.nn.relu,
                elegy.nn.Linear(2),
            ]
        )(jnp.ones([10, 3]))

        elegy.nn.Sequential(
            lambda: [
                elegy.nn.Flatten(),
                elegy.nn.Linear(5),
                jax.nn.relu,
                elegy.nn.Linear(2),
            ]
        )(jnp.ones([10, 3]))

    def test_on_predict(self):

        model = elegy.Model(
            elegy.nn.Sequential(
                lambda: [
                    elegy.nn.Flatten(),
                    elegy.nn.Linear(5),
                    jax.nn.relu,
                    elegy.nn.Linear(2),
                ]
            )
        )

        x = jnp.ones([3, 5])

        y_pred = model.predict(x)
        logs = model.evaluate(x)
