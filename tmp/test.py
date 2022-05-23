from copy import copy
from typing import Generator

import cloudpickle
import jax
import jax.numpy as jnp
import treeo as to

import elegy as eg


def gen(n: int) -> Generator[int, None, None]:
    for i in range(n):
        yield i


def f(g0):
    for epoch in range(2):
        g_iter = copy(g0)

        for i in g_iter:
            print(epoch, i)


f(gen(10))
