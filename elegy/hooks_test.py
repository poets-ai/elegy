import jax
import jax.numpy as jnp

import elegy


def test_losses():
    assert elegy.get_losses() is None

    with elegy.hooks_context():
        elegy.add_loss("x", 2.0)
        losses = elegy.get_losses()

    assert losses["x_loss"] == 2.0


def test_metrics():
    assert elegy.get_metrics() is None

    with elegy.hooks_context():
        elegy.add_metric("x", 2.0)
        metrics = elegy.get_metrics()

    assert metrics["x"] == 2.0


def test_summaries():
    assert elegy.get_summaries() is None

    with elegy.hooks_context():
        elegy.add_summary(("a", 0, "b"), None, 2.0)
        summaries = elegy.get_summaries()

    assert summaries[0] == (("a", 0, "b"), None, 2.0)


def test_jit():
    assert elegy.get_losses() is None

    def f(x):
        x = 2.0 * x
        elegy.add_loss("x", x)
        elegy.add_metric("x", x + 1)
        elegy.add_summary(("a", 0, "b"), jax.nn.relu, x + 2)
        return x

    f_ = elegy.jit(f)

    with elegy.hooks_context():
        x = f_(3.0)
        losses = elegy.get_losses()
        metrics = elegy.get_metrics()
        summaries = elegy.get_summaries()

    assert losses["x_loss"] == 6
    assert metrics["x"] == 7
    assert summaries[0] == (("a", 0, "b"), jax.nn.relu, 8)


def test_value_and_grad():
    assert elegy.get_losses() is None

    def f(x):
        x = 2.0 * x
        elegy.add_loss("x", x)
        elegy.add_metric("x", x + 1)
        elegy.add_summary(("a", 0, "b"), jax.nn.relu, x + 2)
        return x

    f_ = elegy.value_and_grad(f)

    with elegy.hooks_context():
        x, grads = f_(3.0)
        losses = elegy.get_losses()
        metrics = elegy.get_metrics()
        summaries = elegy.get_summaries()

    assert grads == 2.0
    assert losses["x_loss"] == 6
    assert metrics["x"] == 7
    assert summaries[0] == (("a", 0, "b"), jax.nn.relu, 8)
