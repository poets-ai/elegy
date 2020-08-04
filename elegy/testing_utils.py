import functools

from jax import random
import typing as tp


def transform_and_run(
    f: tp.Optional[tp.Callable] = None,
    seed: tp.Optional[int] = 42,
    run_apply: bool = True,
):
    """Transforms the given function and runs init then (optionally) apply.

  Equivalent to:

  >>> def f(x):
  ...   return x
  >>> x = jnp.ones([])
  >>> rng = jax.random.PRNGKey(42)
  >>> f = hk.transform_with_state(f)
  >>> parameters, state = f.init(rng, x)
  >>> out = f.apply(params, state, rng, x)

  This function makes it very convenient to unit test Haiku:

  >>> class MyTest(unittest.TestCase):
  ...   @hk.testing.transform_and_run
  ...   def test_linear_output(self):
  ...     mod = hk.Linear(1)
  ...     out = mod(jnp.ones([1, 1]))
  ...     self.assertEqual(out.ndim, 2)

  And can also be useful in an interactive environment like ipython, Jupyter or
  Google Colaboratory:

  >>> f = lambda x: hk.Bias()(x)
  >>> hk.testing.transform_and_run(f)(jnp.ones([1, 1]))
  DeviceArray([[1.]], dtype=float32)

  See :func:`transform` for more details.

  Args:
    f: A function method to transform.
    seed: A seed to pass to init and apply.
    run_apply: Whether to run apply as well as init. Defaults to true.

  Returns:
    A function that transforms f and runs `init` and optionally `apply`.
  """
    if f is None:
        return functools.partial(transform_and_run, seed=seed, run_apply=run_apply)

    @functools.wraps(f)
    def wrapper(*a, **k):
        """Runs init and apply of f."""
        rng = random.PRNGKey(seed) if seed is not None else None
        transformed = hooks.transform(lambda: f(*a, **k))
        params, state = transformed.init(rng)
        if run_apply:
            transformed_state = transformed.apply(
                params, state, rng, get_summaries=True
            )
            return transformed_state.outputs

    return wrapper
