import typing as tp

import haiku as hk
import jax.numpy as jnp
import numpy as np
from elegy import module
from elegy.initializers import TruncatedNormal, VarianceScaling, Zeros
from elegy.types import Initializer
from jax import lax

PRNGKey = tp.Any
Shape = tp.Tuple[int]
Dtype = tp.Any  # this could be a real type?
Array = tp.Any

default_kernel_init = VarianceScaling()


def _normalize_axes(axes, ndim):
    # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
    return tuple([ax if ax >= 0 else ndim + ax for ax in axes])


class LinearGeneral(module.Module):
    """A linear transformation with flexible axes."""

    features: tp.Union[int, tp.Iterable[int]]
    axis: tp.Union[int, tp.Iterable[int]]
    batch_dims: tp.Iterable[int]
    use_bias: bool
    dtype: Dtype
    kernel_init: tp.Callable
    bias_init: tp.Callable
    precision: tp.Any

    def __init__(
        self,
        features: tp.Union[int, tp.Iterable[int]],
        axis: tp.Union[int, tp.Iterable[int]] = -1,
        batch_dims: tp.Iterable[int] = (),
        use_bias: bool = True,
        dtype: Dtype = jnp.float32,
        kernel_init: tp.Callable = default_kernel_init,
        bias_init: tp.Callable = Zeros,
        precision: tp.Any = None,
    ):
        """Creates a LinearGeneral object.

        Arguments:
            features: int or tuple with number of output features.
            axis: int or tuple with axes to apply the transformation on. For instance,
                (-2, -1) will apply the transformation to the last two axes.
            batch_dims: tuple with batch axes.
            use_bias: whether to add a bias to the output (default: True).
            dtype: the dtype of the computation (default: float32).
            kernel_init: initializer function for the weight matrix.
            bias_init: initializer function for the bias.
            precision: numerical precision of the computation see `jax.lax.Precision`
                for details.
        """
        self.features = features
        self.axis = axis
        self.batch_dims = batch_dims
        self.use_bias = use_bias
        self.dtype = dtype
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.precision = precision

    def __call__(self, inputs: Array) -> Array:
        """Applies a linear transformation to the inputs along multiple dimensions.

        Args:
          inputs: The nd-array to be transformed.

        Returns:
          The transformed input.
        """
        inputs = jnp.asarray(inputs, self.dtype)

        ndim = inputs.ndim
        n_batch_dims = len(self.batch_dims)
        axis = _normalize_axes(self.axis, ndim)
        batch_dims = _normalize_axes(self.batch_dims, ndim)
        n_axis, n_features = len(axis), len(self.features)

        def kernel_init_wrap(rng, shape, dtype=jnp.float32):
            size_batch_dims = np.prod(shape[:n_batch_dims], dtype=np.int32)
            flat_shape = (
                np.prod(shape[n_batch_dims : n_axis + n_batch_dims]),
                np.prod(shape[-n_features:]),
            )
            kernel = jnp.concatenate(
                [
                    self.kernel_init(rng, flat_shape, dtype)
                    for _ in range(size_batch_dims)
                ],
                axis=0,
            )
            return jnp.reshape(kernel, shape)

        batch_shape = tuple([inputs.shape[ax] for ax in batch_dims])
        kernel_shape = tuple([inputs.shape[ax] for ax in axis]) + self.features
        kernel = self.add_parameter(
            name="kernel",
            shape=batch_shape + kernel_shape,
            initializer=kernel_init_wrap,
            trainable=True,
        )
        kernel = jnp.asarray(kernel, self.dtype)

        batch_ind = tuple(range(n_batch_dims))
        contract_ind = tuple(range(n_batch_dims, n_axis + n_batch_dims))
        out = lax.dot_general(
            inputs,
            kernel,
            ((axis, contract_ind), (batch_dims, batch_ind)),
            precision=self.precision,
        )
        if self.use_bias:

            def bias_init_wrap(rng, shape, dtype=jnp.float32):
                size_batch_dims = np.prod(shape[:n_batch_dims], dtype=np.int32)
                flat_shape = (np.prod(shape[-n_features:]),)
                bias = jnp.concatenate(
                    [
                        self.bias_init(rng, flat_shape, dtype)
                        for _ in range(size_batch_dims)
                    ],
                    axis=0,
                )
                return jnp.reshape(bias, shape)

            bias = self.add_parameter(
                name="bias",
                shape=batch_shape + self.features,
                initializer=bias_init_wrap,
                trainable=True,
            )

            # Reshape bias for broadcast.
            expand_dims = sorted(set(range(inputs.ndim)) - set(axis) - set(batch_dims))
            for ax in expand_dims:
                bias = jnp.expand_dims(bias, ax)
            bias = jnp.asarray(bias, self.dtype)
            out = out + bias
        return out


class Linear(module.Module):
    """Linear module."""

    w: np.ndarray
    b: np.ndarray

    def __init__(
        self,
        output_size: int,
        with_bias: bool = True,
        w_init: tp.Optional[Initializer] = None,
        b_init: tp.Optional[Initializer] = None,
        **kwargs
    ):
        """
        Constructs the Linear module.

        Arguments:
            output_size: Output dimensionality.
            with_bias: Whether to add a bias to the output.
            w_init: Optional initializer for weights. By default, uses random values
                from truncated normal, with stddev `1 / sqrt(fan_in)`. See
                https://arxiv.org/abs/1502.03167v3.
            b_init: Optional initializer for bias. By default, zero.
            kwargs: Additional keyword arguments passed to Module.
        """
        super().__init__(**kwargs)
        self.input_size = None
        self.output_size = output_size
        self.with_bias = with_bias
        self.w_init = w_init
        self.b_init = b_init or jnp.zeros

    def call(self, inputs: np.ndarray) -> np.ndarray:
        """"""
        if not inputs.shape:
            raise ValueError("Input must not be scalar.")

        input_size = self.input_size = inputs.shape[-1]
        output_size = self.output_size
        dtype = jnp.float32

        w_init = self.w_init

        if w_init is None:
            stddev = 1.0 / np.sqrt(self.input_size)
            w_init = TruncatedNormal(stddev=stddev)

        w = self.add_parameter(
            "w", [input_size, output_size], dtype, initializer=w_init
        )

        inputs = jnp.asarray(inputs, self.dtype)
        w = jnp.asarray(w, self.dtype)
        out = jnp.dot(inputs, w)

        if self.with_bias:
            b = self.add_parameter(
                "b", [self.output_size], dtype, initializer=self.b_init
            )
            b = jnp.broadcast_to(b, out.shape)
            b = jnp.asarray(b, self.dtype)
            out = out + b

        return out
