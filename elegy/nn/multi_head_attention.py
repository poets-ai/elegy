import typing as tp

import jax
import numpy as np
from jax import numpy as jnp

from elegy import initializers, module
from elegy.nn.dropout import Dropout
from elegy.nn.layer_normalization import LayerNormalization
from elegy.nn.linear import Linear
from elegy.nn.sequential_module import sequential
from elegy import types


class MultiHeadAttention(module.Module):
    r"""
    MultiHead Attention layer.
    Defines the MultiHead Attention operation as described in
    [Attention Is All You Need](https://arxiv.org/abs/1706.03762) which takes
    in the tensors `query`, `key`, and `value`, and returns the dot-product attention
    between them:

    ```python
    mha = MultiHeadAttention(head_size=128, num_heads=12)
    query = tf.random.uniform((32, 20, 200)) # (batch_size, query_elements, query_depth)
    key = tf.random.uniform((32, 15, 300)) # (batch_size, key_elements, key_depth)
    value = tf.random.uniform((32, 15, 400)) # (batch_size, key_elements, value_depth)
    attention = mha([query, key, value]) # (batch_size, query_elements, value_depth)
    ```

    If `value` is not given then internally `value = key` will be used:

    ```python
    mha = MultiHeadAttention(head_size=128, num_heads=12)
    query = tf.random.uniform((32, 20, 200)) # (batch_size, query_elements, query_depth)
    key = tf.random.uniform((32, 15, 300)) # (batch_size, key_elements, key_depth)
    attention = mha([query, key]) # (batch_size, query_elements, key_depth)
    ```

    Arguments:
        head_size: int, dimensionality of the `query`, `key` and `value` tensors
        after the linear transformation.
        num_heads: int, number of attention heads.
        output_size: int, dimensionality of the output space, if `None` then the
        input dimension of
        `value` or `key` will be used, default `None`.
        dropout: float, `rate` parameter for the dropout layer that is
        applied to attention after softmax,
        default `0`.
        use_projection_bias: bool, whether to use a bias term after the linear
        output projection.
        return_attn_coef: bool, if `True`, return the attention coefficients as
        an additional output argument.
        kernel_initializer: initializer, initializer for the kernel weights.
        kernel_regularizer: regularizer, regularizer for the kernel weights.
        kernel_constraint: constraint, constraint for the kernel weights.
        bias_initializer: initializer, initializer for the bias weights.
        bias_regularizer: regularizer, regularizer for the bias weights.
        bias_constraint: constraint, constraint for the bias weights.

    """

    def __init__(
        self,
        head_size: int,
        num_heads: int,
        output_size: tp.Optional[int] = None,
        dropout: float = 0.0,
        use_projection_bias: bool = True,
        return_attn_coef: bool = False,
        kernel_initializer: types.Initializer = initializers.VarianceScaling(scale=2.0),
        bias_initializer: types.Initializer = initializers.Constant(0.0),
        # kernel_initializer: typing.Union[str, typing.Callable] = "glorot_uniform",
        # kernel_regularizer: typing.Union[str, typing.Callable] = None,
        # kernel_constraint: typing.Union[str, typing.Callable] = None,
        # bias_regularizer: typing.Union[str, typing.Callable] = None,
        # bias_constraint: typing.Union[str, typing.Callable] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        if output_size is not None and output_size < 1:
            raise ValueError("output_size must be a positive number")

        self.head_size = head_size
        self.num_heads = num_heads
        self.output_size = output_size
        self.use_projection_bias = use_projection_bias
        self.return_attn_coef = return_attn_coef
        self.droput_rate = dropout

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def call(
        self,
        query: jnp.ndarray,
        key: tp.Optional[jnp.ndarray] = None,
        value: tp.Optional[jnp.ndarray] = None,
        mask=None,
        training=None,
    ):
        """
        Arguments:
            inputs:  List of `[query, key, value]` where
                * `query`: np.ndarray of shape `(..., query_elements, query_depth)`
                * `key`: `np.ndarray of shape '(..., key_elements, key_depth)`
                * `value`: np.ndarray of shape `(..., key_elements, value_depth)`, optional, if not given `key` will be used.
            mask: a binary np.ndarray of shape `[batch_size?, num_heads?, query_elements, key_elements]`
                which specifies which query elements can attendo to which key elements,
                `1` indicates attention and `0` indicates no attention.
        Output shape:
            * `(..., query_elements, output_size)` if `output_size` is given, else
            * `(..., query_elements, value_depth)` if `value` is given, else
            * `(..., query_elements, key_depth)`
        """

        # einsum nomenclature
        # ------------------------
        # N = query elements
        # M = key/value elements
        # H = heads
        # I = input features
        # O = output features

        if key is None:
            key = query

        if value is None:
            value = key

        output_size = (
            self.output_size if self.output_size is not None else value.shape[-1]
        )

        # verify shapes
        if key.shape[-2] != value.shape[-2]:
            raise ValueError(
                "the number of elements in 'key' must be equal to the same as the number of elements in 'value'"
            )

        if mask is not None:
            if len(mask.shape) < 2:
                raise ValueError("'mask' must have atleast 2 dimensions")
            if query.shape[-2] != mask.shape[-2]:
                raise ValueError(
                    "mask's second to last dimension must be equal to the number of elements in 'query'"
                )
            if key.shape[-2] != mask.shape[-1]:
                raise ValueError(
                    "mask's last dimension must be equal to the number of elements in 'key'"
                )

        # get weights
        query_kernel = self.add_parameter(
            "query_kernel",
            [self.num_heads, query.shape[-1], self.head_size],
            jnp.float32,
            initializer=self.kernel_initializer,
        )
        key_kernel = self.add_parameter(
            "key_kernel",
            [self.num_heads, key.shape[-1], self.head_size],
            jnp.float32,
            initializer=self.kernel_initializer,
        )
        value_kernel = self.add_parameter(
            "value_kernel",
            [self.num_heads, value.shape[-1], self.head_size],
            jnp.float32,
            initializer=self.kernel_initializer,
        )
        projection_kernel = self.add_parameter(
            "projection_kernel",
            [self.num_heads, self.head_size, output_size],
            jnp.float32,
            initializer=self.kernel_initializer,
        )

        # Linear transformations
        query = jnp.einsum("...NI , HIO -> ...NHO", query, query_kernel)
        key = jnp.einsum("...MI , HIO -> ...MHO", key, key_kernel)
        value = jnp.einsum("...MI , HIO -> ...MHO", value, value_kernel)

        # Scale dot-product, doing the division to either query or key
        # instead of their product saves some computation
        query /= jnp.sqrt(self.head_size)

        # Calculate dot product attention
        logits = jnp.einsum("...NHO,...MHO->...HNM", query, key)

        # apply mask
        if mask is not None:
            mask = mask.astype(jnp.float32)

            # possibly expand on the head dimension so broadcasting works
            if len(mask.shape) != len(logits.shape):
                mask = jnp.expand_dims(mask, -3)

            logits += -10e9 * (1.0 - mask)

        attn_coef = jax.nn.softmax(logits)

        # attention dropout
        attn_coef_dropout = Dropout(self.droput_rate)(attn_coef, training=training)

        # attention * value
        multihead_output = jnp.einsum("...HNM,...MHI->...NHI", attn_coef_dropout, value)

        # Run the outputs through another linear projection layer. Recombining heads
        # is automatically done.
        output = jnp.einsum("...NHI,HIO->...NO", multihead_output, projection_kernel)

        if self.use_projection_bias:
            output += self.add_parameter(
                "projection_bias",
                [output_size],
                jnp.float32,
                initializer=self.bias_initializer,
            )

        if self.return_attn_coef:
            return output, attn_coef
        else:
            return output
