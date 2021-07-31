import unittest
from tempfile import TemporaryDirectory

import elegy
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import sh
import tensorflow as tf
from elegy.generalized_module.generalized_module import generalize
from flax import linen


class ModuleC(linen.Module):
    @linen.compact
    def __call__(self, x):
        c1 = self.param("c1", lambda _: jnp.ones([5]))
        c2 = self.variable("states", "c2", lambda: jnp.ones([6]))

        return x


class ModuleB(linen.Module):
    @linen.compact
    def __call__(self, x):
        b1 = self.param("b1", lambda _: jnp.ones([3]))
        b2 = self.variable("states", "b2", lambda: jnp.ones([4]))

        x = ModuleC()(x)

        return x


class ModuleA(linen.Module):
    @linen.compact
    def __call__(self, x):
        a1 = self.param("a1", lambda _: jnp.ones([1]))
        a2 = self.variable("states", "a2", lambda: jnp.ones([2]))

        x = ModuleB()(x)

        return x


class TestLinenModule(unittest.TestCase):
    def test_basic(self):
        class M(linen.Module):
            @linen.compact
            def __call__(self, x):

                initialized = self.has_variable("batch_stats", "n")

                vn = self.variable("batch_stats", "n", lambda: 0)

                w = self.param("w", lambda key: 2.0)

                if initialized:
                    vn.value += 1

                return x * w

        gm = generalize(M())
        rng = elegy.RNGSeq(42)

        y_true, params, states = gm.init(rng)(x=3.0, y=1)

        assert y_true == 6
        assert params["w"] == 2
        assert states["batch_stats"]["n"] == 0

        params = params.copy(dict(w=10.0))
        y_true, params, states = gm.apply(params, states, training=True, rng=rng)(
            x=3.0, y=1
        )

        assert y_true == 30
        assert params["w"] == 10
        assert states["batch_stats"]["n"] == 1

    def test_summaries(self):

        model = elegy.Model(ModuleA())

        summary_text = model.summary(x=jnp.ones([10, 2]), depth=1, return_repr=True)
        assert summary_text is not None

        lines = summary_text.split("\n")
        assert "(10, 2)" in lines[3]
        assert "(10, 2)" in lines[5]

        assert "ModuleB_0" in lines[12]
        assert "8" in lines[12]
        assert "32 B" in lines[12]
        assert "10" in lines[12]
        assert "40 B" in lines[12]

        assert "a1" in lines[14]
        assert "1" in lines[14]
        assert "4 B" in lines[14]

        assert "a2" in lines[16]
        assert "2" in lines[16]
        assert "8 B" in lines[16]

        assert "9" in lines[18]
        assert "36 B" in lines[18]
        assert "12" in lines[18]
        assert "48 B" in lines[18]

        assert "21" in lines[21]
        assert "84 B" in lines[21]

    def test_save_flax(self):
        class MLP(linen.Module):
            """Standard LeNet-300-100 MLP network."""

            n1: int = 300
            n2: int = 100

            @linen.compact
            def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
                x = x.astype(jnp.float32) / 255.0

                x = x.reshape(x.shape[0], -1)  # flatten
                x = linen.Dense(self.n1)(x)
                x = jax.nn.relu(x)
                x = linen.Dense(self.n2)(x)
                x = jax.nn.relu(x)
                x = linen.Dense(10)(x)

                return x

        with TemporaryDirectory() as model_dir:

            model = elegy.Model(
                module=MLP(),
                loss=[
                    elegy.losses.MeanSquaredError(),
                    # elegy.regularizers.GlobalL2(l=1e-4),
                ],
                metrics=elegy.metrics.MeanSquaredError(),
                optimizer=optax.adam(1e-3),
            )

            x = np.random.uniform(size=(3000, 6))
            y = np.random.uniform(size=(3000, 10))

            history = model.fit(
                x=x,
                y=y,
                epochs=2,
                steps_per_epoch=3,
                # batch_size=64,
                # validation_data=(x, y),
                # shuffle=True,
                # callbacks=[
                # elegy.callbacks.ModelCheckpoint(
                #     f"{model_dir}_best", save_best_only=True
                # )
                # ],
            )
            model.save(model_dir)

            output = str(sh.ls(model_dir))

            assert "model.pkl" in output

            model = elegy.load(model_dir)

    def test_saved_model_flax(self):

        with TemporaryDirectory() as model_dir:

            model = elegy.Model(
                linen.Dense(1),
                loss=elegy.losses.MeanSquaredError(),
                optimizer=optax.adam(1e-3),
            )

            x = np.random.uniform(size=(3000, 6))
            y = np.random.uniform(size=(3000, 1))

            with pytest.raises(elegy.types.ModelNotInitialized):
                model.saved_model(x, model_dir, batch_size=[1, 2, 4, 8])

            model.fit(x, y, epochs=10)
            model.saved_model(x, model_dir, batch_size=[1, 2, 4, 8])

            output = str(sh.ls(model_dir))

            assert "saved_model.pb" in output
            assert "variables" in output

            saved_model = tf.saved_model.load(model_dir)

            saved_model
