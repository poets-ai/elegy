from elegy.utils import inject_dependencies
import pytest
from unittest import TestCase


class DIFunctionTests(TestCase):
    def test_positional(self):
        def f(a, b, c):
            return a + b + c

        g = inject_dependencies(f)

        y = g("a", "b", "c")

        assert y == "abc"

    def test_positional_error_missing(self):
        def f(a, b, c):
            return a + b + c

        g = inject_dependencies(f)

        with pytest.raises(KeyError):
            g("a", "b")

    def test_positional_error_remaining(self):
        def f(a, b, c):
            return a + b + c

        g = inject_dependencies(f)

        with pytest.raises(TypeError):
            g("a", "b", "c", "d")

    def test_positional_extras_ok(self):
        def f(a, b, c):
            return a + b + c

        g = inject_dependencies(f)

        y = g("a", "b", "c", d="d")

        assert y == "abc"

    def test_keyword(self):
        def f(a, b, c):
            return a + b + c

        g = inject_dependencies(f)

        y = g(b="b", c="c", a="a")

        assert y == "abc"

    def test_keyword_extras_ok(self):
        def f(a, b, c):
            return a + b + c

        g = inject_dependencies(f)

        y = g(b="b", c="c", a="a", d="d")

        assert y == "abc"

    def test_keyword_error_missing(self):
        def f(a, b, c):
            return a + b + c

        g = inject_dependencies(f)

        with pytest.raises(KeyError):
            g(b="b", c="c")

    def test_mixed(self):
        def f(a, b, c):
            return a + b + c

        g = inject_dependencies(f)

        y = g("a", c="c", b="b")

        assert y == "abc"

    def test_mixed_ignore_duplicated_kwarg_in_arg(self):
        def f(a, b, c):
            return a + b + c

        g = inject_dependencies(f)

        y = g("a", c="c", b="b", a="f")

        assert y == "abc"

    def test_override_defaults(self):
        def f(a, b, c="x"):
            return a + b + c

        g = inject_dependencies(f)

        y = g("a", c="c", b="b")

        assert y == "abc"
