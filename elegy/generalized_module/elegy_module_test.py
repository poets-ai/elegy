from tempfile import TemporaryDirectory
from unittest import TestCase

import elegy
import jax
import jax.numpy as jnp
import numpy as np
import optax
import sh
from elegy.generalized_module.generalized_module import generalize


class ModuleC(elegy.Module):
    def call(self, x):
        c1 = self.add_parameter("c1", lambda: jnp.ones([5]))
        c2 = self.add_parameter("c2", lambda: jnp.ones([6]), trainable=False)

        x = jax.nn.relu(x)
        self.add_summary("relu", jax.nn.relu, x)

        return x


class ModuleB(elegy.Module):
    def call(self, x):
        b1 = self.add_parameter("b1", lambda: jnp.ones([3]))
        b2 = self.add_parameter("b2", lambda: jnp.ones([4]), trainable=False)

        x = ModuleC()(x)

        x = jax.nn.relu(x)
        self.add_summary("relu", jax.nn.relu, x)

        return x


class ModuleA(elegy.Module):
    def call(self, x):
        a1 = self.add_parameter("a1", lambda: jnp.ones([1]))
        a2 = self.add_parameter("a2", lambda: jnp.ones([2]), trainable=False)

        x = ModuleB()(x)

        x = jax.nn.relu(x)
        self.add_summary("relu", jax.nn.relu, x)

        return x


class ElegyModuleTest(TestCase):
    def test_basic(self):
        class M(elegy.Module):
            def call(self, x):
                n = self.add_parameter("n", lambda: 0, trainable=False)
                w = self.add_parameter("w", lambda: 2.0)

                self.update_parameter("n", n + 1)

                key = self.next_key()

                return x * w

        gm = generalize(M())
        rng = elegy.RNGSeq(42)

        y_true, params, states = gm.init(rng)(x=3.0, y=1)

        assert y_true == 6
        assert params["w"] == 2
        assert states["states"]["n"] == 0

        params["w"] = 10.0
        y_true, params, states = gm.apply(params, states, training=True, rng=rng)(
            x=3.0, y=1
        )

        assert y_true == 30
        assert params["w"] == 10
        assert states["states"]["n"] == 1

    def test_summaries(self):

        model = elegy.Model(ModuleA())

        x = jnp.ones([10, 2])

        # model.init(x=x)

        summary_text = model.summary(x, depth=1, return_repr=True)
        assert summary_text is not None

        lines = summary_text.split("\n")

        assert "module_b" in lines[7]
        assert "ModuleB" in lines[7]
        assert "(10, 2)" in lines[7]
        assert "8" in lines[7]
        assert "32 B" in lines[7]
        assert "10" in lines[7]
        assert "40 B" in lines[7]

        assert "relu" in lines[9]
        assert "(10, 2)" in lines[9]

        assert "*" in lines[11]
        assert "ModuleA" in lines[11]
        assert "(10, 2)" in lines[11]
        assert "1" in lines[11]
        assert "4 B" in lines[11]
        assert "2" in lines[11]
        assert "8 B" in lines[11]

        assert "9" in lines[13]
        assert "36 B" in lines[13]

        assert "12" in lines[13]
        assert "48 B" in lines[13]

        assert "21" in lines[16]
        assert "84 B" in lines[16]

    def test_save(self):
        class MLP(elegy.Module):
            def call(self, x: jnp.ndarray) -> jnp.ndarray:
                x = x.astype(jnp.float32) / 255.0

                x = x.reshape(x.shape[0], -1)  # flatten
                x = elegy.nn.Linear(10)(x)
                x = jax.nn.relu(x)
                x = elegy.nn.Linear(10)(x)
                x = jax.nn.relu(x)
                x = elegy.nn.Linear(10)(x)

                return x

        with TemporaryDirectory() as model_dir:

            model = elegy.Model(
                module=MLP(),
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
