from elegy import utils
import pytest
from unittest import TestCase


class DIFunctionTests(TestCase):
    def test_positional(self):
        def f(a, b, c):
            return a + b + c

        g = utils.inject_dependencies(f)

        y = g("a", "b", "c")

        assert y == "abc"

    def test_positional_error_missing(self):
        def f(a, b, c):
            return a + b + c

        g = utils.inject_dependencies(f)

        with pytest.raises(TypeError):
            g("a", "b")

    def test_positional_error_remaining(self):
        def f(a, b, c):
            return a + b + c

        g = utils.inject_dependencies(f)

        with pytest.raises(TypeError):
            g("a", "b", "c", "d")

    def test_positional_extras_ok(self):
        def f(a, b, c):
            return a + b + c

        g = utils.inject_dependencies(f)

        y = g("a", "b", "c", d="d")

        assert y == "abc"

    def test_keyword(self):
        def f(a, b, c):
            return a + b + c

        g = utils.inject_dependencies(f)

        y = g(b="b", c="c", a="a")

        assert y == "abc"

    def test_keyword_extras_ok(self):
        def f(a, b, c):
            return a + b + c

        g = utils.inject_dependencies(f)

        y = g(b="b", c="c", a="a", d="d")

        assert y == "abc"

    def test_keyword_error_missing(self):
        def f(a, b, c):
            return a + b + c

        g = utils.inject_dependencies(f)

        with pytest.raises(TypeError):
            g(b="b", c="c")

    def test_mixed(self):
        def f(a, b, c):
            return a + b + c

        g = utils.inject_dependencies(f)

        y = g("a", c="c", b="b")

        assert y == "abc"

    def test_mixed_ignore_duplicated_kwarg_in_arg(self):
        def f(a, b, c):
            return a + b + c

        g = utils.inject_dependencies(f)

        y = g("a", c="c", b="b", a="f")

        assert y == "abc"

    def test_override_defaults(self):
        def f(a, b, c="x"):
            return a + b + c

        g = utils.inject_dependencies(f)

        y = g("a", c="c", b="b")

        assert y == "abc"


class SplitTest(TestCase):
    def test_basic(self):
        from deepmerge import always_merger
        import toolz

        d = {"a/n/m": {"x": 1, "y/hola": 2}, "a/n/m/x/t": 10, "b": {"z": 3, "k": 5}}
        ds = list(utils.split(d))

        dn = toolz.reduce(always_merger.merge, ds, {})

        print(ds)
        print(dn)
