from elegy.dependency_injection import DIFunction
import pytest


def test_positional():
    def f(a, b, c):
        return a + b + c

    g = DIFunction.create(f)

    y = g("a", "b", "c")

    assert y == "abc"


def test_positional_error_missing():
    def f(a, b, c):
        return a + b + c

    g = DIFunction.create(f)

    with pytest.raises(KeyError):
        g("a", "b")


def test_positional_error_remaining():
    def f(a, b, c):
        return a + b + c

    g = DIFunction.create(f)

    with pytest.raises(TypeError):
        g("a", "b", "c", "d")


def test_positional_extras_ok():
    def f(a, b, c):
        return a + b + c

    g = DIFunction.create(f)

    y = g("a", "b", "c", d="d")

    assert y == "abc"


def test_keyword():
    def f(a, b, c):
        return a + b + c

    g = DIFunction.create(f)

    y = g(b="b", c="c", a="a")

    assert y == "abc"


def test_keyword_extras_ok():
    def f(a, b, c):
        return a + b + c

    g = DIFunction.create(f)

    y = g(b="b", c="c", a="a", d="d")

    assert y == "abc"


def test_keyword_error_missing():
    def f(a, b, c):
        return a + b + c

    g = DIFunction.create(f)

    with pytest.raises(KeyError):
        g(b="b", c="c")


def test_mixed():
    def f(a, b, c):
        return a + b + c

    g = DIFunction.create(f)

    y = g("a", c="c", b="b")

    assert y == "abc"


def test_mixed_ignore_duplicated_kwarg_in_arg():
    def f(a, b, c):
        return a + b + c

    g = DIFunction.create(f)

    y = g("a", c="c", b="b", a="f")

    assert y == "abc"


def test_override_defaults():
    def f(a, b, c="x"):
        return a + b + c

    g = DIFunction.create(f)

    y = g("a", c="c", b="b")

    assert y == "abc"
