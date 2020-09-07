from operator import mod
import threading
import typing as tp
from contextlib import contextmanager
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from elegy import module
from elegy.module import Module, get_unique_name
from elegy.random import RNG
from elegy.types import PRNGKey
from elegy.utils import EMPTY, Empty
