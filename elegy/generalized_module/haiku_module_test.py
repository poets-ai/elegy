import unittest
from tempfile import TemporaryDirectory

import elegy
import haiku
import jax
import jax.numpy as jnp
import numpy as np
import optax
import sh
from elegy.generalized_module.generalized_module import generalize


class ModuleC(haiku.Module):
    def __call__(self, x):
        c1 = haiku.get_parameter("c1", [5], jnp.int32, init=jnp.ones)
        c2 = haiku.get_state("c2", [6], jnp.int32, init=jnp.ones)

        x = jax.nn.relu(x)

        return x


class ModuleB(haiku.Module):
    def __call__(self, x):
        b1 = haiku.get_parameter("b1", [3], jnp.int32, init=jnp.ones)
        b2 = haiku.get_state("b2", [4], jnp.int32, init=jnp.ones)

        x = ModuleC()(x)

        x = jax.nn.relu(x)

        return x


class ModuleA(haiku.Module):
    def __call__(self, x):
        a1 = haiku.get_parameter("a1", [1], jnp.int32, init=jnp.ones)
        a2 = haiku.get_state("a2", [2], jnp.int32, init=jnp.ones)

        x = ModuleB()(x)

        x = jax.nn.relu(x)

        return x


class TestHaikuModule(unittest.TestCase):
    def test_basic(self):
        class M(haiku.Module):
            def __call__(self, x):

                n = haiku.get_state(
                    "n", shape=[], dtype=jnp.int32, init=lambda *args: np.array(0)
                )
                w = haiku.get_parameter("w", [], init=lambda *args: np.array(2.0))

                haiku.set_state("n", n + 1)

                return x * w

        def f(x, initializing, rng):
            return M()(x)

        gm = elegy.HaikuModule(f)
        rng = elegy.RNGSeq(42)

        y_true, params, states = gm.init(rng)(x=3.0, y=1, rng=None, initializing=True)

        assert y_true == 6
        assert params["m"]["w"] == 2
        assert states["m"]["n"] == 0

        params = haiku.data_structures.to_mutable_dict(params)
        params["m"]["w"] = np.array(10.0)
        y_true, params, states = gm.apply(params, states, training=True, rng=rng)(
            x=3.0, y=1, rng=None, initializing=True
        )

        assert y_true == 30
        assert params["m"]["w"] == 10
        assert states["m"]["n"] == 1

    def test_summaries(self):
        def f(x):
            return ModuleA()(x)

        model = elegy.Model(elegy.HaikuModule(f))

        summary_text = model.summary(x=jnp.ones([10, 2]), depth=2, return_repr=True)
        assert summary_text is not None

    def test_save(self):
        class MLP(haiku.Module):
            def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
                x = x.astype(jnp.float32) / 255.0

                x = x.reshape(x.shape[0], -1)  # flatten
                x = haiku.Linear(10)(x)
                x = jax.nn.relu(x)
                x = haiku.Linear(10)(x)
                x = jax.nn.relu(x)
                x = haiku.Linear(10)(x)

                return x

        with TemporaryDirectory() as model_dir:

            model = elegy.Model(
                module=elegy.HaikuModule(lambda x: MLP()(x)),
                loss=[
                    elegy.losses.MeanSquaredError(),
                    elegy.regularizers.GlobalL2(l=1e-4),
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
                batch_size=64,
                validation_data=(x, y),
                shuffle=True,
                callbacks=[
                    elegy.callbacks.ModelCheckpoint(
                        f"{model_dir}_best", save_best_only=True
                    )
                ],
            )
            model.save(model_dir)

            output = str(sh.ls(model_dir))

            assert "model.pkl" in output

            model = elegy.load(model_dir)
