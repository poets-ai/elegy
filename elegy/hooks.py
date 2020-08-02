# Some portiong of this code are adapted from Haiku:
# https://github.com/deepmind/dm-haiku/blob/master/haiku/_src/transform.py#L228#L300

import threading
import typing as tp
from contextlib import contextmanager
from dataclasses import dataclass

import haiku as hk
import numpy as np

from haiku._src import transform as src_transform


T = tp.TypeVar("T")
LOCAL = threading.local()
LOCAL.contexts = []

