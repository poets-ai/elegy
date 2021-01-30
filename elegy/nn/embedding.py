# File ported from: https://raw.githubusercontent.com/deepmind/dm-haiku/master/haiku/_src/embed.py

"""Modules for performing embedding lookups in Haiku."""

import enum
from typing import Optional, Union

import jax
import jax.numpy as jnp

from elegy.module import Module
from elegy import initializers, types


class EmbedLookupStyle(enum.Enum):
    """How to return the embedding matrices given IDs."""

    ARRAY_INDEX = 1
    ONE_HOT = 2


# Needed for members to show in sphinx docs.
for style in EmbedLookupStyle:
    style.__doc__ = style.name


class Embedding(Module):
    """Module for embedding tokens in a low-dimensional space."""

    def __init__(
        self,
        vocab_size: Optional[int] = None,
        embed_dim: Optional[int] = None,
        embedding_matrix: Optional[jnp.ndarray] = None,
        w_init: Optional[types.Initializer] = None,
        lookup_style: Union[str, EmbedLookupStyle] = "ARRAY_INDEX",
        name: Optional[str] = None,
    ):
        """
        Constructs an Embed module.

        Args:
            vocab_size: The number of unique tokens to embed. If not provided, an
                existing vocabulary matrix from which ``vocab_size`` can be inferred
                must be provided as ``existing_vocab``.
            embed_dim: Number of dimensions to assign to each embedding. If an
                existing vocabulary matrix initializes the module, this should not be
                provided as it will be inferred.
            embedding_matrix: A matrix-like object equivalent in size to
                ``[vocab_size, embed_dim]``. If given, it is used as the initial value
                for the embedding matrix and neither ``vocab_size`` or ``embed_dim``
                need be given. If they are given, their values are checked to be
                consistent with the dimensions of ``embedding_matrix``.
            w_init: An initializer for the embeddings matrix. As a default,
                embeddings are initialized via a truncated normal distribution.
            lookup_style: One of the enum values of :class:`EmbedLookupStyle`
                determining how to access the value of the embeddings given an ID.
                Regardless the input should be a dense array of integer values
                representing ids. This setting changes how internally this module maps
                those ides to embeddings. The result is the same, but the speed and
                memory tradeoffs are different. It default to using numpy-style array
                indexing. This value is only the default for the module, and at any
                given invocation can be overridden in :meth:`__call__`.
            name: Optional name for this module.

        Raises:
            ValueError: If none of ``embed_dim``, ``embedding_matrix`` and
                ``vocab_size`` are supplied, or if ``embedding_matrix`` is supplied
                and ``embed_dim`` or ``vocab_size`` is not consistent with the
                supplied matrix.
        """
        super().__init__(name=name)
        if embedding_matrix is None and not (vocab_size and embed_dim):
            raise ValueError(
                "Embedding must be supplied either with an initial `embedding_matrix` "
                "or with `embed_dim` and `vocab_size`."
            )
        if embedding_matrix is not None:
            embedding_matrix = jnp.asarray(embedding_matrix)
            if vocab_size and embedding_matrix.shape[0] != vocab_size:
                raise ValueError(
                    "An `embedding_matrix` was supplied but the `vocab_size` of "
                    f"{vocab_size} was not consistent with its shape "
                    f"{embedding_matrix.shape}."
                )
            if embed_dim and embedding_matrix.shape[1] != embed_dim:
                raise ValueError(
                    "An `embedding_matrix` was supplied but the `embed_dim` of "
                    f"{embed_dim} was not consistent with its shape "
                    f"{embedding_matrix.shape}."
                )
            self.embeddings = self.add_parameter(
                "embeddings",
                embedding_matrix.shape,
                initializer=lambda _, __: embedding_matrix,
            )
        else:
            assert embed_dim is not None
            assert vocab_size is not None

            w_init = w_init or initializers.TruncatedNormal()
            self.embeddings = self.add_parameter(
                "embeddings", [vocab_size, embed_dim], initializer=w_init
            )

        self.vocab_size = vocab_size or embedding_matrix.shape[0]
        self.embed_dim = embed_dim or embedding_matrix.shape[1]
        self.lookup_style = lookup_style

    def call(
        self,
        ids: jnp.ndarray,
        lookup_style: Optional[Union[str, EmbedLookupStyle]] = None,
    ) -> jnp.ndarray:
        r"""
        Lookup embeddings.

        Looks up an embedding vector for each value in ``ids``. All ids must be
        within ``[0, vocab_size)`` to prevent ``NaN``\ s from propagating.

        Args:
            ids: integer array.
            lookup_style: Overrides the ``lookup_style`` given in the constructor.

        Returns:
            Tensor of ``ids.shape + [embedding_dim]``.

        Raises:
            AttributeError: If ``lookup_style`` is not valid.
            ValueError: If ``ids`` is not an integer array.
        """
        # TODO(tomhennigan) Consider removing asarray here.
        ids = jnp.asarray(ids)
        if not jnp.issubdtype(ids.dtype, jnp.integer):
            raise ValueError(
                "Embedding's __call__ method must take an array of "
                "integer dtype but was called with an array of "
                f"{ids.dtype}"
            )

        lookup_style = lookup_style or self.lookup_style
        if isinstance(lookup_style, str):
            lookup_style = getattr(EmbedLookupStyle, lookup_style.upper())

        if lookup_style == EmbedLookupStyle.ARRAY_INDEX:
            return self.embeddings[(ids,)]

        elif lookup_style == EmbedLookupStyle.ONE_HOT:
            one_hot_ids = jax.nn.one_hot(ids, self.vocab_size)[..., None]
            return (self.embeddings * one_hot_ids).sum(axis=-2)

        else:
            raise NotImplementedError(f"{lookup_style} is not supported by Embedding.")
