import typing as tp

import jax
import numpy as np
from jax import numpy as jnp

from elegy import initializers, module, hooks
from elegy.nn.dropout import Dropout
from elegy.nn.layer_normalization import LayerNormalization
from elegy.nn.linear import Linear
from elegy.nn.sequential import sequential


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
    Call Arguments:
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

    def __init__(
        self,
        head_size: int,
        num_heads: int,
        output_size: tp.Optional[int] = None,
        dropout: float = 0.0,
        use_projection_bias: bool = True,
        return_attn_coef: bool = False,
        kernel_initializer: initializers.Initializer = initializers.VarianceScaling(
            scale=2.0
        ),
        bias_initializer: initializers.Initializer = initializers.Constant(0.0),
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
        query_kernel = hooks.get_parameter(
            "query_kernel",
            [self.num_heads, query.shape[-1], self.head_size],
            jnp.float32,
            initializer=self.kernel_initializer,
        )
        key_kernel = hooks.get_parameter(
            "key_kernel",
            [self.num_heads, key.shape[-1], self.head_size],
            jnp.float32,
            initializer=self.kernel_initializer,
        )
        value_kernel = hooks.get_parameter(
            "value_kernel",
            [self.num_heads, value.shape[-1], self.head_size],
            jnp.float32,
            initializer=self.kernel_initializer,
        )
        projection_kernel = hooks.get_parameter(
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
            output += hooks.get_parameter(
                "projection_bias",
                [output_size],
                jnp.float32,
                initializer=self.bias_initializer,
            )

        if self.return_attn_coef:
            return output, attn_coef
        else:
            return output


class TransformerEncoderLayer(module.Module):
    r"""
    TransformerEncoderLayer is made up of self-attn and feedforward network.
    
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        head_size: the number of expected features in the input (required).
        num_heads: the number of heads in the multiheadattention models (required).
        output_size: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(head_size=512, num_heads=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(
        self,
        head_size: int,
        num_heads: int,
        output_size: tp.Optional[int] = None,
        dropout: float = 0.0,
        activation=jax.nn.relu,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.head_size = head_size
        self.num_heads = num_heads
        self.output_size = output_size
        self.dropout = dropout
        self.activation = activation

    def call(
        self,
        src: np.ndarray,
        mask: tp.Optional[np.ndarray] = None,
        # src_key_padding_mask: tp.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # Implementation of Feedforward model

        output_size: int = (
            self.output_size if self.output_size is not None else src.shape[-1]
        )

        src2 = MultiHeadAttention(self.head_size, self.num_heads, dropout=self.dropout)(
            src, mask=mask
        )
        src = src + Dropout(self.dropout)(src2)
        src = LayerNormalization()(src)
        src2 = sequential(
            Linear(output_size),
            self.activation,
            Dropout(self.dropout),
            Linear(output_size),
        )(src)
        src = src + Dropout(self.dropout)(src2)
        src = LayerNormalization()(src)
        return src


class TransformerEncoder(module.Module):
    r"""
    TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> transformer_encoder = elegy.nn.TransformerEncoder(
                lambda: elegy.nn.TransformerEncoderLayer(head_size=512, num_heads=8), 
                num_layers=6,
            )
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    def __init__(
        self,
        encoder_layer: tp.Callable[[], module.Module],
        num_layers: int,
        norm: tp.Optional[tp.Callable[[], module.Module]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder_layer = encoder_layer
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: np.ndarray,
        mask: tp.Optional[np.ndarray] = None,
        # src_key_padding_mask: tp.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for _ in range(self.num_layers):
            output = self.encoder_layer()(output, mask=mask)

        if self.norm is not None:
            output = self.norm()(output)

        return output


class TransformerDecoderLayer(module.Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        head_size: the number of expected features in the input (required).
        num_heads: the number of heads in the multiheadattention models (required).
        output_size: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(head_size=512, num_heads=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(
        self,
        head_size: int,
        num_heads: int,
        output_size: int = 2048,
        dropout: float = 0.1,
        activation: tp.Callable[[np.ndarray], np.ndarray] = jax.nn.relu,
    ):
        super().__init__()
        self.head_size = head_size
        self.num_heads = num_heads
        self.output_size = output_size
        self.dropout = dropout
        self.activation = activation

    def call(
        self,
        tgt: np.ndarray,
        memory: np.ndarray,
        tgt_mask: tp.Optional[np.ndarray] = None,
        memory_mask: tp.Optional[np.ndarray] = None,
        # tgt_key_padding_mask: tp.Optional[np.ndarray] = None,
        # memory_key_padding_mask: tp.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # Implementation of Feedforward model

        tgt2 = MultiHeadAttention(self.head_size, self.num_heads, dropout=self.dropout)(
            tgt, mask=tgt_mask
        )
        tgt = tgt + Dropout(self.dropout)(tgt2)
        tgt = LayerNormalization()(tgt)
        tgt2 = MultiHeadAttention(self.head_size, self.num_heads, dropout=self.dropout)(
            tgt, memory, mask=memory_mask,
        )
        tgt = tgt + Dropout(self.dropout)(tgt2)
        tgt = LayerNormalization()(tgt)
        tgt = tgt + sequential(
            Linear(self.output_size),
            self.activation,
            Dropout(self.dropout),
            Linear(self.head_size),
            Dropout(self.dropout),
        )(tgt)
        tgt = LayerNormalization()(tgt)
        return tgt


class TransformerDecoder(module.Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(head_size=512, num_heads=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """

    def __init__(
        self,
        decoder_layer: tp.Callable[[], module.Module],
        num_layers: int,
        norm: tp.Optional[tp.Callable[[], module.Module]] = None,
    ):
        super().__init__()
        self.decoder_layer = decoder_layer
        self.num_layers = num_layers
        self.norm = norm

    def call(
        self,
        tgt: np.ndarray,
        memory: np.ndarray,
        tgt_mask: tp.Optional[np.ndarray] = None,
        memory_mask: tp.Optional[np.ndarray] = None,
        # tgt_key_padding_mask: tp.Optional[np.ndarray] = None,
        # memory_key_padding_mask: tp.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for _ in range(self.num_layers):
            output = self.decoder_layer()(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                # tgt_key_padding_mask=tgt_key_padding_mask,
                # memory_key_padding_mask=memory_key_padding_mask,
            )

        if self.norm is not None:
            output = self.norm()(output)

        return output


class Transformer(module.Module):
    r"""A transformer model. User is able to modify the attributes as needed. The architecture
    is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
    Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
    Processing Systems, pages 6000-6010. Users can build the BERT(https://arxiv.org/abs/1810.04805)
    model with corresponding parameters.

    Args:
        head_size: the number of expected features in the encoder/decoder inputs (default=512).
        num_heads: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        output_size: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, relu or gelu (default=relu).
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).

    Examples::
        >>> transformer_model = nn.Transformer(num_heads=16, num_encoder_layers=12)
        >>> src = torch.rand((10, 32, 512))
        >>> tgt = torch.rand((20, 32, 512))
        >>> out = transformer_model(src, tgt)

    Note: A full example to apply nn.Transformer module for the word language model is available in
    https://github.com/pytorch/examples/tree/master/word_language_model
    """

    def __init__(
        self,
        head_size: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        output_size: int = 2048,
        dropout: float = 0.1,
        activation: tp.Callable[[np.ndarray], np.ndarray] = jax.nn.relu,
        custom_encoder: tp.Optional[tp.Any] = None,
        custom_decoder: tp.Optional[tp.Any] = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.head_size = head_size
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.output_size = output_size
        self.dropout = dropout
        self.activation = activation
        self.custom_encoder = custom_encoder
        self.custom_decoder = custom_decoder

    def forward(
        self,
        src: np.ndarray,
        tgt: np.ndarray,
        src_mask: tp.Optional[np.ndarray] = None,
        tgt_mask: tp.Optional[np.ndarray] = None,
        memory_mask: tp.Optional[np.ndarray] = None,
        # src_key_padding_mask: tp.Optional[np.ndarray] = None,
        # tgt_key_padding_mask: tp.Optional[np.ndarray] = None,
        # memory_key_padding_mask: tp.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        r"""Take in and process masked source/target sequences.

        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
            src_mask: the additive mask for the src sequence (optional).
            tgt_mask: the additive mask for the tgt sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
            src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).
            tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).

        Shape:
            - src: :math:`(S, N, E)`.
            - tgt: :math:`(T, N, E)`.
            - src_mask: :math:`(S, S)`.
            - tgt_mask: :math:`(T, T)`.
            - memory_mask: :math:`(T, S)`.
            - src_key_padding_mask: :math:`(N, S)`.
            - tgt_key_padding_mask: :math:`(N, T)`.
            - memory_key_padding_mask: :math:`(N, S)`.

            Note: [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight. 
            [src/tgt/memory]_key_padding_mask provides specified elements in the key to be ignored by
            the attention. If a ByteTensor is provided, the non-zero positions will be ignored while the zero
            positions will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.

            - output: :math:`(T, N, E)`.

            Note: Due to the multi-head attention architecture in the transformer model,
            the output sequence length of a transformer is same as the input sequence
            (i.e. target) length of the decode.

            where S is the source sequence length, T is the target sequence length, N is the
            batch size, E is the feature number

        Examples:
            >>> output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        """

        if src.shape[1] != tgt.shape[1]:
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.shape[2] != self.head_size or tgt.shape[2] != self.head_size:
            raise RuntimeError(
                "the feature number of src and tgt must be equal to head_size"
            )

        if self.custom_encoder is not None:
            encoder = self.custom_encoder()
        else:
            encoder = TransformerEncoder(
                lambda: TransformerEncoderLayer(
                    self.head_size,
                    self.num_heads,
                    self.output_size,
                    self.dropout,
                    self.activation,
                ),
                self.num_encoder_layers,
                lambda: LayerNormalization(self.head_size),
            )

        if self.custom_decoder is not None:
            decoder = self.custom_decoder()
        else:
            decoder = TransformerDecoder(
                lambda: TransformerDecoderLayer(
                    self.head_size,
                    self.num_heads,
                    self.output_size,
                    self.dropout,
                    self.activation,
                ),
                self.num_decoder_layers,
                lambda: LayerNormalization(self.head_size),
            )

        memory = encoder(
            src,
            mask=src_mask,
            # src_key_padding_mask=src_key_padding_mask
        )
        output = decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            # tgt_key_padding_mask=tgt_key_padding_mask,
            # memory_key_padding_mask=memory_key_padding_mask,
        )

        return output
