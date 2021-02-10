import typing as tp

import jax
import jax.numpy as jnp
import numpy as np

from elegy.module import Module
from elegy.nn.dropout import Dropout
from elegy.nn.layer_normalization import LayerNormalization
from elegy.nn.linear import Linear
from elegy.nn.multi_head_attention import MultiHeadAttention
from elegy.nn.sequential_module import sequential


class TransformerEncoderLayer(Module):
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
        >>> # encoder_layer = nn.TransformerEncoderLayer(head_size=512, num_heads=8)
        >>> # src = torch.rand(10, 32, 512)
        >>> # out = encoder_layer(src)
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

        src2 = MultiHeadAttention(
            self.head_size,
            self.num_heads,
            dropout=self.dropout,
        )(src, mask=mask)
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


class TransformerEncoder(Module):
    r"""
    TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> # transformer_encoder = elegy.nn.TransformerEncoder(
                lambda: elegy.nn.TransformerEncoderLayer(head_size=512, num_heads=8),
                num_layers=6,
            )
        >>> # src = torch.rand(10, 32, 512)
        >>> # out = transformer_encoder(src)
    """

    def __init__(
        self,
        encoder_layer: tp.Callable[[], Module],
        num_layers: int,
        norm: tp.Optional[tp.Callable[[], Module]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder_layer = encoder_layer
        self.num_layers = num_layers
        self.norm = norm

    def call(
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


class TransformerDecoderLayer(Module):
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
        >>> # decoder_layer = nn.TransformerDecoderLayer(head_size=512, num_heads=8)
        >>> # memory = torch.rand(10, 32, 512)
        >>> # tgt = torch.rand(20, 32, 512)
        >>> # out = decoder_layer(tgt, memory)
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
            tgt,
            memory,
            mask=memory_mask,
        )
        tgt = tgt + Dropout(self.dropout)(tgt2)
        tgt = LayerNormalization()(tgt)
        tgt = tgt + sequential(
            Linear(self.output_size),
            self.activation,
            Dropout(self.dropout),
            Linear(self.output_size),
            Dropout(self.dropout),
        )(tgt)
        tgt = LayerNormalization()(tgt)
        return tgt


class TransformerDecoder(Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> # decoder_layer = nn.TransformerDecoderLayer(head_size=512, num_heads=8)
        >>> # transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> # memory = torch.rand(10, 32, 512)
        >>> # tgt = torch.rand(20, 32, 512)
        >>> # out = transformer_decoder(tgt, memory)
    """

    def __init__(
        self,
        decoder_layer: tp.Callable[[], Module],
        num_layers: int,
        norm: tp.Optional[tp.Callable[[], Module]] = None,
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


class Transformer(Module):
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
        >>> # transformer_model = nn.Transformer(num_heads=16, num_encoder_layers=12)
        >>> # src = torch.rand((10, 32, 512))
        >>> # tgt = torch.rand((20, 32, 512))
        >>> # out = transformer_model(src, tgt)

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

    def call(
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
            >>> # output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        """

        if src.shape[0] != tgt.shape[0]:
            raise RuntimeError("the batch number of src and tgt must be equal")

        # if src.shape[2] != self.head_size or tgt.shape[2] != self.head_size:
        #     raise RuntimeError(
        #         "the feature number of src and tgt must be equal to head_size"
        #     )

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
                lambda: LayerNormalization(),
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
                lambda: LayerNormalization(),
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
