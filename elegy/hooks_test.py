import jax
import jax.numpy as jnp
import unittest
import pytest

import elegy


class TestHooks(unittest.TestCase):
    def test_losses(self):
        assert elegy.get_losses() is None

        with elegy.hooks_context():
            elegy.add_loss("x", 2.0)
            losses = elegy.get_losses()

        assert losses["x_loss"] == 2.0

    def test_metrics(self):
        assert elegy.get_metrics() is None

        with elegy.hooks_context():
            elegy.add_metric("x", 2.0)
            metrics = elegy.get_metrics()

        assert metrics["x"] == 2.0

    def test_summaries(self):
        assert elegy.get_summaries() is None

        with elegy.hooks_context(summaries=True):
            elegy.add_summary(("a", 0, "b"), None, 2.0)
            summaries = elegy.get_summaries()

        assert summaries[0] == (("a", 0, "b"), None, 2.0)

    def test_no_summaries(self):
        assert elegy.get_summaries() is None

        with elegy.hooks_context(summaries=False):
            elegy.add_summary(("a", 0, "b"), None, 2.0)
            summaries = elegy.get_summaries()

        assert summaries is None

    def test_rng(self):
        assert elegy.get_rng() is None
        rng = elegy.RNGSeq(42)
        initial_key = rng.key

        with pytest.raises(ValueError):
            elegy.next_key()

        with elegy.hooks_context(rng=rng):
            key = elegy.next_key()
            new_rng = elegy.get_rng()

        assert jnp.alltrue(initial_key != key)
        assert jnp.alltrue(initial_key != new_rng.key)

    def test_training(self):
        assert elegy.get_training() is None

        with pytest.raises(ValueError):
            elegy.is_training()

        with elegy.hooks_context(training=False):
            training = elegy.is_training()

        assert training == False

    def test_jit(self):
        assert elegy.get_losses() is None
        rng = elegy.RNGSeq(42)
        initial_key = rng.key

        def f(x):
            x = 2.0 * x
            elegy.add_loss("x", x)
            elegy.add_metric("x", x + 1)
            elegy.add_summary(("a", 0, "b"), jax.nn.relu, x + 2)
            key = elegy.next_key()
            training_f = elegy.get_training()

            assert isinstance(training_f, bool)

            return x, key, training_f

        f_ = elegy.jit(f)

        with elegy.hooks_context(summaries=True, rng=rng, training=False):
            x, key, training_f = f_(3.0)
            losses = elegy.get_losses()
            metrics = elegy.get_metrics()
            summaries = elegy.get_summaries()
            new_rng = elegy.get_rng()
            training = elegy.is_training()

        assert x == 6
        assert losses["x_loss"] == 6
        assert metrics["x"] == 7
        assert summaries[0] == (("a", 0, "b"), jax.nn.relu, 8)
        assert jnp.alltrue(initial_key != key)
        assert jnp.alltrue(initial_key != new_rng.key)
        assert training == False
        assert training_f == False

    def test_value_and_grad(self):
        assert elegy.get_losses() is None
        rng = elegy.RNGSeq(42)
        initial_key = rng.key

        def f(x):
            x = 2.0 * x
            elegy.add_loss("x", x)
            elegy.add_metric("x", x + 1)
            elegy.add_summary(("a", 0, "b"), jax.nn.relu, x + 2)
            key = elegy.next_key()
            training_f = elegy.get_training()

            assert isinstance(training_f, bool)

            return x, key, training_f

        f_ = elegy.value_and_grad(f, has_aux=True)

        with elegy.hooks_context(summaries=True, rng=rng, training=False):
            (x, key, training_f), grads = f_(3.0)
            losses = elegy.get_losses()
            metrics = elegy.get_metrics()
            summaries = elegy.get_summaries()
            new_rng = elegy.get_rng()
            training = elegy.is_training()

        assert x == 6
        assert grads == 2.0
        assert losses["x_loss"] == 6
        assert metrics["x"] == 7
        assert summaries[0] == (("a", 0, "b"), jax.nn.relu, 8)
        assert jnp.alltrue(initial_key != key)
        assert jnp.alltrue(initial_key != new_rng.key)
        assert training == False
        assert training_f == False
