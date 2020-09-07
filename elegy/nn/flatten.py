from elegy.module import Module
import typing as tp

import haiku as hk
import jax.numpy as jnp
import numpy as np


from elegy import types


def _infer_shape(output_shape, dimensions):
    """
    Replaces the -1 wildcard in the output shape vector.

    This function infers the correct output shape given the input dimensions.

    Args:
        output_shape: Output shape.
        dimensions: List of input non-batch dimensions.

    Returns:
        Tuple of non-batch output dimensions.
    """
    # Size of input.
    n = np.prod(dimensions)
    # Size of output where defined.
    v = np.array(output_shape)
    m = abs(np.prod(v))
    # Replace wildcard.
    v[v == -1] = n // m
    return tuple(v)


class Reshape(Module):
    """
    Reshapes input Tensor, preserving the batch dimension.

    For example, given an input tensor with shape `[B, H, W, C, D]`:

    ```python
    B, H, W, C, D = range(1, 6)
    x = jnp.ones([B, H, W, C, D])
    ```

    The default behavior when `output_shape` is `(-1, D)` is to flatten
    all dimensions between `B` and `D`:

    ```python
    mod = elegy.nn.Reshape(output_shape=(-1, D))
    assert mod(x).shape == (B, H*W*C, D)
    ```

    You can change the number of preserved leading dimensions via
    `preserve_dims`:

    ```python
    mod = elegy.nn.Reshape(output_shape=(-1, D), preserve_dims=2)
    assert mod(x).shape == (B, H, W*C, D)
    mod = elegy.nn.Reshape(output_shape=(-1, D), preserve_dims=3)
    assert mod(x).shape == (B, H, W, C, D)
    mod = elegy.nn.Reshape(output_shape=(-1, D), preserve_dims=4)
    assert mod(x).shape == (B, H, W, C, 1, D)
    ```
    """

    def __init__(self, output_shape: types.Shape, preserve_dims: int = 1, **kwargs):
        """
        Constructs a `Reshape` module.

        Args:
            output_shape: Shape to reshape the input tensor to while preserving its
                first `preserve_dims` dimensions. When the special value -1 appears in
                `output_shape` the corresponding size is automatically inferred. Note
                that -1 can only appear once in `output_shape`.
                To flatten all non-batch dimensions use `Flatten`.
            preserve_dims: Number of leading dimensions that will not be reshaped.
            kwargs: Additional keyword arguments passed to Module.

        Raises:
            ValueError: If `preserve_dims` is not positive.
        """
        super().__init__(**kwargs)
        if preserve_dims <= 0:
            raise ValueError("Argument preserve_dims should be >= 1.")
        if output_shape.count(-1) > 1:
            raise ValueError("-1 can only occur once in `output_shape`.")

        self.output_shape = tuple(output_shape)
        self.preserve_dims = preserve_dims

    def call(self, inputs: np.ndarray) -> np.ndarray:
        """
        Arguments:
            inputs: the array to be reshaped.

        Returns:
            A reshaped array.
        """
        if inputs.ndim <= self.preserve_dims:
            return inputs

        if -1 in self.output_shape:
            reshaped_shape = _infer_shape(
                self.output_shape, inputs.shape[self.preserve_dims :]
            )
        else:
            reshaped_shape = self.output_shape
        shape = inputs.shape[: self.preserve_dims] + reshaped_shape
        return jnp.reshape(inputs, shape)


class Flatten(Reshape):
    """
    Flattens the input, preserving the batch dimension(s).

    By default, Flatten combines all dimensions except the first.
    Additional leading dimensions can be preserved by setting preserve_dims.

    ```python
    x = jnp.ones([3, 2, 4])
    flat = elegy.nn.Flatten()
    assert flat(x).shape == (3, 8)
    ```

    When the input to flatten has fewer than `preserve_dims` dimensions it is
    returned unchanged:

    ```python
    x = jnp.ones([3])
    assert flat(x).shape == (3,)
    ```
    """

    def __init__(
        self,
        preserve_dims: int = 1,
        name: tp.Optional[str] = None,
    ):
        super().__init__(output_shape=(-1,), preserve_dims=preserve_dims, name=name)
