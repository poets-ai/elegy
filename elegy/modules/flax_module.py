import typing as tp

import flax.linen as nn
import jax
import jax.numpy as jnp
import treeo as to
from attr import mutable
from flax.core.frozen_dict import FrozenDict

from elegy import types, utils

FrozerVariables = FrozenDict[str, tp.Mapping[str, tp.Any]]
Variables = tp.Mapping[str, tp.Mapping[str, tp.Any]]
M = tp.TypeVar("M", bound="nn.module.Module")
RNGDict = tp.Dict[str, jnp.ndarray]


class ModuleState(tp.Generic[M], to.Tree, to.Immutable):
    variables: tp.Optional[FrozerVariables] = to.node()

    hashable_module: to.Hashable[M] = to.static()
    mutable_train: tp.Tuple[str] = to.static()
    mutable_eval: tp.Tuple[str] = to.static()
    rngs_init: tp.Tuple[str] = to.static()
    rngs_apply: tp.Tuple[str] = to.static()
    method_init: str = to.static()

    def __init__(
        self,
        module: M,
        variables: tp.Optional[Variables] = None,
        mutable_train: tp.Sequence[str] = ("batch_stats", "cache"),
        mutable_eval: tp.Optional[tp.Sequence[str]] = None,
        rngs_init: tp.Sequence[str] = ("params",),
        rngs_apply: tp.Sequence[str] = ("dropout",),
        method_init: str = "__call__",
    ):

        self.hashable_module = to.Hashable(module)
        self.variables = FrozenDict(variables) if variables is not None else None
        self.mutable_train = tuple(mutable_train)
        self.mutable_eval = (
            tuple(mutable_eval) if mutable_eval is not None else tuple(mutable_train)
        )
        self.rngs_init = tuple(rngs_init)
        self.rngs_apply = tuple(rngs_apply)
        self.method_init = method_init

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

        module_manager: ModuleState[M] = self

        if "method" not in kwargs:
            method = getattr(module_manager.module, self.method_init)
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

        variables = module_manager.module.init(
            rngs,
            *args,
            method=method,
            **kwargs,
        )

        if not isinstance(variables, FrozenDict):
            variables = FrozenDict(variables)

        module_manager = module_manager.replace(
            variables=variables,
            hashable_module=to.Hashable(module_manager.module),
        )

        return module_manager

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
