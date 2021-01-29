import jax
import jax.numpy as jnp
import unittest
import pytest

import elegy


class TestHooks(unittest.TestCase):
    def test_losses(self):
        assert not elegy.hooks.losses_active()

        with elegy.context(set_all=True):
            elegy.add_loss("x", 2.0)
            losses = elegy.get_losses()

        assert losses["x_loss"] == 2.0

    def test_metrics(self):
        assert not elegy.hooks.metrics_active()

        with elegy.context(set_all=True):
            elegy.add_metric("x", 2.0)
            metrics = elegy.get_metrics()

        assert metrics["x"] == 2.0

    def test_summaries(self):
        assert not elegy.hooks.summaries_active()

        with elegy.context(summaries=True):
            elegy.add_summary(("a", 0, "b"), None, 2.0)
            summaries = elegy.get_summaries()

        assert summaries[0] == (("a", 0, "b"), None, 2.0)

    def test_no_summaries(self):
        assert not elegy.hooks.summaries_active()

        with elegy.context(summaries=False):
            elegy.add_summary(("a", 0, "b"), None, 2.0)
            has_summaries = elegy.hooks.summaries_active()

        assert not has_summaries

    def test_jit(self):
        assert not elegy.hooks.losses_active()

        def f(x):
            x = 2.0 * x
            elegy.add_loss("x", x)
            elegy.add_metric("x", x + 1)
            elegy.add_summary(("a", 0, "b"), jax.nn.relu, x + 2)

            return x

        f_ = elegy.jit(f)

        with elegy.context(set_all=True):
            x = f_(3.0)
            losses = elegy.get_losses()
            metrics = elegy.get_metrics()
            summaries = elegy.get_summaries()

        assert x == 6
        assert losses["x_loss"] == 6
        assert metrics["x"] == 7
        assert summaries[0] == (("a", 0, "b"), jax.nn.relu, 8)
