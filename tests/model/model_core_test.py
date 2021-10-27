import typing as tp
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import treex as tx

import elegy
from elegy.model.model_core import ModelCore


class ModelCoreTest(unittest.TestCase):
    def test_init(self):
        N = 0

        class Model(ModelCore):
            a: jnp.ndarray = tx.node()

            def __init__(self):
                super().__init__()
                self.a = jnp.array(-1, dtype=jnp.int32)

            def init_step(
                self,
                key: jnp.ndarray,
                inputs: tp.Any,
            ) -> "Model":
                nonlocal N

                N += 1
                self.a = jnp.array(0, dtype=jnp.int32)
                print("JITTING")
                return self

        model = Model()
        inputs = np.array(1.0)

        assert N == 0
        assert model.a == -1

        model.init_on_batch(inputs)
        assert N == 1

        # jits again because _initialized changed
        model.init_on_batch(inputs)
        assert N == 2

        # no jit change this time
        model.init_on_batch(inputs)
        assert N == 2

    def test_pred_step(self):
        N = 0

        class Model(ModelCore):
            a: jnp.ndarray = tx.node()

            def init_step(
                self,
                key: jnp.ndarray,
                inputs: tp.Any,
            ) -> "Model":
                self.a = jnp.array(0, dtype=jnp.int32)
                return self

            def pred_step(self, inputs):
                nonlocal N
                N += 1

                self.a += 1

                return 1, self

        model = Model()

        preds = model.predict_on_batch(inputs=np.array(1.0))
        assert N == 1
        assert preds == 1
        assert model.a == 1

        preds = model.predict_on_batch(inputs=np.array(1.0))
        assert N == 1
        assert preds == 1
        assert model.a == 2

        model.eager = True

        preds = model.predict_on_batch(inputs=(np.array(1.0)))
        assert N == 2
        assert preds == 1
        assert model.a == 3

    def test_test_step(self):
        N = 0

        class Model(ModelCore):
            a: jnp.ndarray = tx.node()

            def init_step(
                self,
                key: jnp.ndarray,
                inputs: tp.Any,
            ) -> "Model":
                self.a = jnp.array(0, dtype=jnp.int32)
                return self

            def test_step(self, inputs, labels) -> elegy.TestStep["Model"]:
                nonlocal N
                N += 1
                self.a += 1
                loss = 1.0

                return loss, dict(loss=loss), self

        model = Model()

        logs = model.test_on_batch(inputs=(np.array(1.0)), labels=(1.0,))
        assert N == 1
        assert logs["loss"] == 1.0
        assert model.a == 1

        logs = model.test_on_batch(inputs=(np.array(1.0)), labels=(1.0,))
        assert N == 1
        assert logs["loss"] == 1.0
        assert model.a == 2

        model.eager = True

        logs = model.test_on_batch(inputs=(np.array(1.0)), labels=(1.0,))
        assert N == 2
        assert logs["loss"] == 1
        assert model.a == 3

    def test_train_step(self):
        N = 0

        class Model(ModelCore):
            a: jnp.ndarray = tx.node()

            def init_step(
                self,
                key: jnp.ndarray,
                inputs: tp.Any,
            ) -> "Model":
                self.a = jnp.array(0, dtype=jnp.int32)
                return self

            def train_step(self, inputs, labels) -> elegy.TrainStep["Model"]:
                nonlocal N
                N += 1
                self.a += 1
                loss = 2.0

                return dict(loss=loss), self

        model = Model()

        logs = model.train_on_batch(inputs=(np.array(1.0)), labels=(1.0,))
        assert N == 1
        assert logs["loss"] == 2.0
        assert model.a == 1

        logs = model.train_on_batch(inputs=(np.array(1.0)), labels=(1.0,))
        assert N == 1
        assert logs["loss"] == 2.0
        assert model.a == 2

        model.eager = True

        logs = model.train_on_batch(inputs=(np.array(1.0)), labels=(1.0,))
        assert N == 2
        assert logs["loss"] == 2.0
        assert model.a == 3
