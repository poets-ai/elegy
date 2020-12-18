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
    """
    Just your regular densely-connected NN layer.

    `Linear` implements the operation:

    ```python
    output = activation(dot(input, kernel) + bias)
    ```

    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).

    Note: If the input to the layer has a rank greater than 2, then `Linear`
    computes the dot product between the `inputs` and the `kernel` along the
    last axis of the `inputs` and axis 1 of the `kernel` (using `tf.tensordot`).
    For example, if input has dimensions `(batch_size, d0, d1)`,
    then we create a `kernel` with shape `(d1, units)`, and the `kernel` operates
    along axis 2 of the `input`, on every sub-tensor of shape `(1, 1, d1)`
    (there are `batch_size * d0` such sub-tensors).

    The output in this case will have shape `(batch_size, d0, units)`.
    Besides, layer attributes cannot be modified after the layer has been called
    once (except the `trainable` attribute).

    Example:

    >>> # Create a `Sequential` model and add a Linear layer as the first layer.
    >>> model = tf.keras.models.Sequential()
    >>> model.add(tf.keras.Input(shape=(16,)))
    >>> model.add(tf.keras.layers.Linear(32, activation='relu'))
    >>> # Now the model will take as input arrays of shape (None, 16)
    >>> # and output arrays of shape (None, 32).
    >>> # Note that after the first layer, you don't need to specify
    >>> # the size of the input anymore:
    >>> model.add(tf.keras.layers.Linear(32))
    >>> model.output_shape
    >>> (None, 32)
    """

    w: np.ndarray
    b: np.ndarray

    def __init__(
        self,
        units: int,
        use_bias: bool = True,
        kernel_initializer: Initializer = VarianceScaling(),
        bias_initializer: Initializer = Zeros(),
        kernel_regularizer: tp.Optional[tp.Callable] = None,
        bias_regularizer: tp.Optional[tp.Callable] = None,
        activity_regularizer: tp.Optional[tp.Callable] = None,
        kernel_constraint: tp.Optional[tp.Callable] = None,
        bias_constraint: tp.Optional[tp.Callable] = None,
        precision: tp.Any = None,
        **kwargs,
    ):
        """

        Arguments:
            units: Positive integer, dimensionality of the output space.
            activation: Activation function to use.
                If you don't specify anything, no activation is applied
                (ie. "linear" activation: `a(x) = x`).
            use_bias: Boolean, whether the layer uses a bias vector.
            kernel_initializer: Initializer for the `kernel` weights matrix.
            bias_initializer: Initializer for the bias vector.
            kernel_regularizer: Regularizer function applied to
                the `kernel` weights matrix.
            bias_regularizer: Regularizer function applied to the bias vector.
            activity_regularizer: Regularizer function applied to
                the output of the layer (its "activation").
            kernel_constraint: Constraint function applied to
                the `kernel` weights matrix.
            bias_constraint: Constraint function applied to the bias vector.
            precision: numerical precision of the computation see `jax.lax.Precision`
                for details.

        Input shape:
            N-D tensor with shape: `(batch_size, ..., input_dim)`.
            The most common situation would be
            a 2D input with shape `(batch_size, input_dim)`.

        Output shape:
            N-D tensor with shape: `(batch_size, ..., units)`.
            For instance, for a 2D input with shape `(batch_size, input_dim)`,
            the output would have shape `(batch_size, units)`.
        """
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)

        # self.input_size = None
        # self.output_size = output_size
        # self.with_bias = with_bias
        # self.w_init = w_init
        # self.b_init = b_init or jnp.zeros

        self.units = units
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.precision = precision

    def call(self, inputs: np.ndarray) -> np.ndarray:
        """"""
        if not inputs.shape:
            raise ValueError("Input must not be scalar.")

        inputs = jnp.asarray(inputs, self.dtype)

        kernel = self.add_parameter(
            name="kernel",
            shape=(inputs.shape[-1], self.units),
            initializer=self.kernel_initializer,
            trainable=True,
        )
        kernel = jnp.asarray(kernel, self.dtype)

        y = lax.dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )
        if self.use_bias:
            bias = self.add_parameter(
                name="bias",
                shape=(self.units,),
                initializer=self.bias_initializer,
                trainable=True,
            )
            bias = jnp.asarray(bias, self.dtype)
            y = y + bias

        return y
