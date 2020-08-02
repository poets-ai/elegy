# Implementation based on Tensorflow Keras and Haiku
# Tensorflow: https://github.com/tensorflow/tensorflow/blob/2b96f3662bd776e277f86997659e61046b56c315/tensorflow/python/keras/layers/normalization.py#L46
# Haiku: https://github.com/deepmind/dm-haiku/blob/master/haiku/_src/batch_norm.py#L39#L194

from elegy import hooks
import typing as tp

import haiku as hk
import numpy as np


class BatchNormalization(hk.BatchNorm):
    r"""
    Normalize and create_scale inputs or activations. (Ioffe and Szegedy, 2014).

    Normalize the activations of the previous layer at each batch,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    
    ### Normalization equations

    Consider the intermediate activations \(x\) of a mini-batch of size
    \\(m\\):
    We can compute the mean and variance of the batch
    
    \\({\mu_B} = \frac{1}{m} \sum_{i=1}^{m} {x_i}\\)
    \\({\sigma_B^2} = \frac{1}{m} \sum_{i=1}^{m} ({x_i} - {\mu_B})^2\\)
    
    and then compute a normalized \\(x\\), including a small factor
    \\({\eps}\\) for numerical stability.
    
    \\(\hat{x_i} = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \eps}}\\)
    
    And finally \\(\hat{x}\\) is linearly transformed by \\({\gamma}\\)
    and \\({\beta}\\), which are learned parameters:
    
    \\({y_i} = {\gamma * \hat{x_i} + \beta}\\)
    
    ### References

    * [Batch Normalization: Accelerating Deep Network Training by Reducing
        Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
    """

    def __init__(
        self,
        create_scale: bool = True,
        create_offset: bool = True,
        decay_rate: float = 0.99,
        eps: float = 1e-5,
        scale_init: tp.Optional[hk.initializers.Initializer] = None,
        offset_init: tp.Optional[hk.initializers.Initializer] = None,
        axis: tp.Optional[tp.Sequence[int]] = None,
        cross_replica_axis: tp.Optional[str] = None,
        data_format: str = "channels_last",
        name: tp.Optional[str] = None,
    ):
        """
        Creates a BatchNormalization instance.
        
        Arguments:
            create_scale: If True, multiply by `gamma`.
                If False, `gamma` is not used.
                When the next layer is linear (also e.g. `relu`),
                this can be disabled since the scaling
                will be done by the next layer.
            create_offset: If True, add offset of `beta` to normalized tensor.
                se, `beta` is ignored.
            axis: Integer, the axis that should be normalized
                (typically the features axis).
                For instance, after a `Conv2D` layer with
                `data_format="channels_first"`,
                set `axis=1` in `BatchNormalization`.
            decay_rate: Momentum parameter for the moving average.
            eps: Small float added to variance to avoid dividing by zero.
            scale_init: Optional initializer for gain (aka create_scale). Can only be set
                if ``create_scale=True``. By default, ``1``.
            offset_init: Optional initializer for bias (aka offset). Can only be set
                if ``create_offset=True``. By default, ``0``.
            axis: Which axes to reduce over. The default (``None``) signifies that all
                but the channel axis should be normalized. Otherwise this is a list of
                axis indices which will have normalization statistics calculated.
            cross_replica_axis: If not ``None``, it should be a string representing
                the axis name over which this module is being run within a ``jax.pmap``.
                Supplying this argument means that batch statistics are calculated
                across all replicas on that axis.
            data_format: The data format of the input. Can be either
                ``channels_first``, ``channels_last``, ``N...C`` or ``NC...``. By
                default it is ``channels_last``.
            name: The module name.
        """
        super().__init__(
            create_scale=create_scale,
            create_offset=create_offset,
            decay_rate=decay_rate,
            eps=eps,
            scale_init=scale_init,
            offset_init=offset_init,
            axis=axis,
            cross_replica_axis=cross_replica_axis,
            data_format=data_format,
            name=name,
        )

    def __call__(
        self,
        inputs: np.ndarray,
        is_training: bool,
        test_local_stats: bool = False,
        scale: tp.Optional[np.ndarray] = None,
        offset: tp.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Computes the normalized version of the input.

        Arguments:
            inputs: An array, where the data format is ``[..., C]``.
            is_training: Whether training is currently happening.
            test_local_stats: Whether local stats are used when is_training=False.
            scale: An array up to n-D. The shape of this tensor must be broadcastable
                to the shape of ``inputs``. This is the scale applied to the normalized
                inputs. This cannot be passed in if the module was constructed with
                ``create_scale=True``.
            offset: An array up to n-D. The shape of this tensor must be broadcastable
                to the shape of ``inputs``. This is the offset applied to the normalized
                inputs. This cannot be passed in if the module was constructed with
                ``create_offset=True``.

        Returns:
            The array, normalized across all but the last dimension.
        """

        outputs = super().__call__(
            inputs=inputs,
            is_training=is_training,
            test_local_stats=test_local_stats,
            scale=scale,
            offset=offset,
        )

        hooks.add_summary(None, self.__class__.__name__, outputs)

        return outputs
