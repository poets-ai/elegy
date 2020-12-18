import typing as tp
from elegy.utils import Protocol


class Constraint(Protocol):
    def __call__(self, w: tp.Any) -> tp.Any:
        ...
