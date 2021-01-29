from unittest import TestCase

import elegy
import jax.numpy as jnp
import pytest


class LossTest(TestCase):
    def test_basic(self):
        class MAE(elegy.metrics.Mean):
            def call(self, y_true, y_pred):
                return super().call(jnp.abs(y_true - y_pred))

        y_true = jnp.array([1.0, 2.0, 3.0])
        y_pred = jnp.array([2.0, 3.0, 4.0])

        mae = MAE()

        loss = mae.call_with_defaults()(y_true, y_pred)
        assert jnp.allclose(loss, 1)

        y_pred = jnp.array([3.0, 4.0, 5.0])

        loss = mae.call_with_defaults()(y_true, y_pred)
        assert jnp.allclose(loss, 1.5)

    def test_slice(self):
        class MAE(elegy.metrics.Mean):
            def call(self, y_true, y_pred):
                return super().call(jnp.abs(y_true - y_pred))

        y_true = dict(a=jnp.array([1.0, 2.0, 3.0]))
        y_pred = dict(a=jnp.array([2.0, 3.0, 4.0]))

        mae = MAE(on="a")

        # raises because it doesn't use kwargs
        with pytest.raises(BaseException):
            sample_loss = mae.call_with_defaults()(y_true, y_pred)

        # raises because it doesn't use __call__ which filters
        with pytest.raises(BaseException):
            sample_loss = mae.call(y_true=y_true, y_pred=y_pred)

        loss = mae.call_with_defaults()(y_true=y_true, y_pred=y_pred)
        assert jnp.allclose(loss, 1)

        y_pred = dict(a=jnp.array([3.0, 4.0, 5.0]))

        loss = mae.call_with_defaults()(y_true=y_true, y_pred=y_pred)
        assert jnp.allclose(loss, 1.5)
