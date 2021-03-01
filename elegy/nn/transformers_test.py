from unittest import TestCase

import elegy
import jax
import jax.numpy as jnp
import numpy as np


class TransformerTest(TestCase):
    def test_connects(self):
        transformer_model = elegy.nn.Transformer(
            head_size=64,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
        )

        src = np.random.uniform(size=(5, 32, 64))
        tgt = np.random.uniform(size=(5, 32, 64))

        _, params, collections = transformer_model.init(rng=elegy.RNGSeq(42))(src, tgt)
        out, params, collections = transformer_model.apply(
            params, collections, rng=elegy.RNGSeq(420)
        )(src, tgt)
