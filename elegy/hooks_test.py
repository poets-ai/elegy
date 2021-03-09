import jax
import jax.numpy as jnp
import unittest
import pytest

import elegy


class TestHooks(unittest.TestCase):
    def test_losses(self):
        assert not elegy.hooks.losses_active()

        with elegy.hooks.context(set_all=True):
            elegy.hooks.add_loss("x", 2.0)
            losses = elegy.hooks.get_losses()

        assert losses["x_loss"] == 2.0

    def test_metrics(self):
        assert not elegy.hooks.metrics_active()

        with elegy.hooks.context(set_all=True):
            elegy.hooks.add_metric("x", 2.0)
            metrics = elegy.hooks.get_metrics()

        assert metrics["x"] == 2.0

    def test_summaries(self):
        assert not elegy.hooks.summaries_active()

        with elegy.hooks.context(summaries=True):
            elegy.hooks.add_summary(("a", 0, "b"), None, 2.0)
            summaries = elegy.hooks.get_summaries()

        assert summaries[0] == (("a", 0, "b"), None, 2.0)

    def test_no_summaries(self):
        assert not elegy.hooks.summaries_active()

        with elegy.hooks.context(summaries=False):
            elegy.hooks.add_summary(("a", 0, "b"), None, 2.0)
            has_summaries = elegy.hooks.summaries_active()

        assert not has_summaries

    def test_jit(self):
        assert not elegy.hooks.losses_active()

        def f(x):
            x = 2.0 * x
            elegy.hooks.add_loss("x", x)
            elegy.hooks.add_metric("x", x + 1)
            elegy.hooks.add_summary(("a", 0, "b"), jax.nn.relu, x + 2)

            return x

        f_ = elegy.hooks.jit(f)

        with elegy.hooks.context(set_all=True):
            x = f_(3.0)
            losses = elegy.hooks.get_losses()
            metrics = elegy.hooks.get_metrics()
            summaries = elegy.hooks.get_summaries()

        assert x == 6
        assert losses["x_loss"] == 6
        assert metrics["x"] == 7
        assert summaries[0] == (("a", 0, "b"), jax.nn.relu, 8)

    def test_named_call(self):
        class Module0(elegy.Module):
            def call(self, x):
                x = elegy.nn.Linear(5)(x)
                x = elegy.nn.Linear(7)(x)
                return x

        m = elegy.Model(Module0())
        m.init(jnp.ones(4))

        with elegy.hooks.context(named_call=True):
            jaxpr = jax.make_jaxpr(
                lambda x, states: m.pred_step(x, states, False, False)
            )(jnp.ones([4]), m.states)
            print(jaxpr)

            assert jaxpr.jaxpr.eqns[0].params["name"] == ()
            assert jaxpr.jaxpr.eqns[0].params["call_jaxpr"].eqns[0].params["name"] == (
                "linear",
            )
            assert jaxpr.jaxpr.eqns[0].params["call_jaxpr"].eqns[1].params["name"] == (
                "linear_1",
            )

        # no named call without hook
        jaxpr = jax.make_jaxpr(lambda x, states: m.pred_step(x, states, False, False))(
            jnp.ones([4]), m.states
        )
        assert jaxpr.jaxpr.eqns[0].primitive.name != "named_call"
