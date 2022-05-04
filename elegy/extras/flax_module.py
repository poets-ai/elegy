import typing as tp

import flax.linen as nn
import jax
import jax.numpy as jnp
import treeo as to
from attr import mutable
from flax import struct
from flax.core.frozen_dict import FrozenDict

from elegy import types, utils

FrozerVariables = FrozenDict[str, tp.Mapping[str, tp.Any]]
Variables = tp.Mapping[str, tp.Mapping[str, tp.Any]]
M = tp.TypeVar("M", bound="nn.module.Module")
RNGDict = tp.Dict[str, jnp.ndarray]
A = tp.TypeVar("A")


# ---------------------------------------------------
# utils
# ---------------------------------------------------
class _Immutable:
    def replace(self: A, **kwargs) -> A:
        raise NotImplementedError()


@struct.dataclass
class ModuleState(tp.Generic[M], _Immutable):
    variables: tp.Optional[FrozerVariables] = struct.field()

    hashable_module: to.Hashable[M] = struct.field(pytree_node=False)
    mutable_train: tp.Tuple[str] = struct.field(pytree_node=False)
    mutable_eval: tp.Tuple[str] = struct.field(pytree_node=False)
    rngs_init: tp.Tuple[str] = struct.field(pytree_node=False)
    rngs_apply: tp.Tuple[str] = struct.field(pytree_node=False)
    method_init: str = struct.field(pytree_node=False)

    @classmethod
    def new(
        cls,
        module: M,
        variables: tp.Optional[Variables] = None,
        mutable_train: tp.Sequence[str] = ("batch_stats", "cache"),
        mutable_eval: tp.Optional[tp.Sequence[str]] = None,
        rngs_init: tp.Sequence[str] = ("params",),
        rngs_apply: tp.Sequence[str] = ("dropout",),
        method_init: str = "__call__",
    ):

        return cls(
            hashable_module=to.Hashable(module),
            variables=FrozenDict(variables) if variables is not None else None,
            mutable_train=tuple(mutable_train),
            mutable_eval=(
                tuple(mutable_eval)
                if mutable_eval is not None
                else tuple(mutable_train)
            ),
            rngs_init=tuple(rngs_init),
            rngs_apply=tuple(rngs_apply),
            method_init=method_init,
        )

    @property
    def module(self: "ModuleState[M]") -> M:
        return self.hashable_module.value

    @property
    def initialized(self: "ModuleState[M]") -> bool:
        return self.variables is not None

    def init(
        self: "ModuleState[M]",
        key: tp.Optional[types.KeyLike],
        *args,
        training: bool = True,
        rngs: tp.Optional[tp.Union[RNGDict, tp.Sequence[str]]] = None,
        **kwargs,
    ) -> "ModuleState[M]":
        """
        Initialize the module.
        """

        if "method" not in kwargs:
            method = getattr(self.module, self.method_init)
        else:
            method = kwargs.pop("method")

        arg_names = utils._function_argument_names(method)

        if arg_names is not None and "training" in arg_names:
            kwargs["training"] = training

        # calculate rngs
        if isinstance(rngs, dict):
            pass
        else:
            if key is not None:
                key = utils.Key(key)

                if rngs is None:
                    rng_collections = self.rngs_init
                else:
                    rng_collections = rngs

                rngs = _split_into_collection(key, rng_collections)
            else:
                rngs = {}

        variables = self.module.init(
            rngs,
            *args,
            method=method,
            **kwargs,
        )

        if not isinstance(variables, FrozenDict):
            variables = FrozenDict(variables)

        self = self.replace(
            variables=variables,
            hashable_module=to.Hashable(self.module),
        )

        return self

    def apply(
        self: "ModuleState[M]",
        key: tp.Optional[types.KeyLike],
        *args,
        training: bool = True,
        rngs: tp.Optional[tp.Union[RNGDict, tp.Sequence[str]]] = None,
        mutable: tp.Optional[tp.Union[bool, tp.Sequence[str]]] = None,
        **kwargs,
    ) -> tp.Tuple[tp.Any, "ModuleState[M]"]:
        return self._forward(
            self.module.__call__,
            key,
            *args,
            training=training,
            rngs=rngs,
            mutable=mutable,
            **kwargs,
        )

    def stateless(
        self: "ModuleState[M]",
        key: tp.Optional[types.KeyLike],
        *args,
        training: bool = True,
        rngs: tp.Optional[tp.Union[RNGDict, tp.Sequence[str]]] = None,
        mutable: tp.Optional[tp.Union[bool, tp.Sequence[str]]] = None,
        **kwargs,
    ) -> tp.Any:
        return self.apply(
            key,
            *args,
            training=training,
            rngs=rngs,
            mutable=mutable,
            **kwargs,
        )[0]

    def _forward(
        self: "ModuleState[M]",
        method: tp.Callable,
        key: tp.Optional[types.KeyLike],
        *args,
        training: bool = True,
        rngs: tp.Optional[tp.Union[RNGDict, tp.Sequence[str]]] = None,
        mutable: tp.Optional[tp.Union[bool, tp.Sequence[str]]] = None,
        **kwargs,
    ) -> tp.Tuple[tp.Any, "ModuleState[M]"]:

        variables = self.variables

        if variables is None:
            raise ValueError(f"'variables' field is not set for module: {self.module}")

        arg_names = utils._function_argument_names(method)

        if arg_names is not None and "training" in arg_names:
            kwargs["training"] = training

        # calculate rngs
        if isinstance(rngs, dict):
            pass
        else:
            if key is not None:
                key = utils.Key(key)

                if rngs is None:
                    rng_collections = self.rngs_apply
                else:
                    rng_collections = rngs

                rngs = _split_into_collection(key, rng_collections)
            else:
                rngs = {}

        # calculate mutable
        if mutable is None:
            mutable = self.mutable_train if training else self.mutable_eval

        output, updates = self.module.apply(
            variables,
            *args,
            rngs=rngs,
            method=method,
            mutable=mutable,
            **kwargs,
        )

        self = self.replace(
            variables=variables.copy(updates),
        )

        return output, self

    def __getitem__(self, key: str) -> tp.Any:
        if self.variables is None:
            raise KeyError(f"'variables' field is not set for module: {self.module}")

        return self.variables[key]

    def __contains__(self, key: str) -> bool:
        if self.variables is None:
            raise KeyError(f"'variables' field is not set for module: {self.module}")

        return key in self.variables

    def update(self: "ModuleState[M]", **kwargs) -> "ModuleState[M]":
        if self.variables is None:
            raise ValueError(f"'variables' field is not set for module: {self.module}")

        return self.replace(variables=self.variables.copy(kwargs))


def _split_into_collection(
    key: jnp.ndarray,
    collection: tp.Sequence[str],
) -> tp.Dict[str, jnp.ndarray]:
    """
    Split the key into the specified rngs.
    """

    rngs = jax.random.split(key, len(collection))

    keys_collection = {col: rngs[i] for i, col in enumerate(collection)}

    return keys_collection
