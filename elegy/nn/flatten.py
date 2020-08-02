import typing as tp


import numpy as np


class Flatten(hk.Flatten):
    """
    Flattens the input, preserving the batch dimension(s).

    By default, Flatten combines all dimensions except the first.
    Additional leading dimensions can be preserved by setting preserve_dims.

    ```python
    x = jnp.ones([3, 2, 4])
    flat = hk.Flatten()
    flat(x).shape # (3, 8)
    ```

    When the input to flatten has fewer than `preserve_dims` dimensions it is
    returned unchanged:

    ```python
    x = jnp.ones([3])
    flat(x).shape # (3,)
    ```
  """

    def __init__(
        self, preserve_dims: int = 1, name: tp.Optional[str] = None,
    ):
        """
        Creates a Flatten instance.

        Arguments:
            preserve_dims: Number of leading dimensions that will not be reshaped.
            name: Name of the module.
        """
        super().__init__(preserve_dims=preserve_dims, name=name)

    def __call__(self, inputs: np.ndarray):
        """
        Arguments:
            inputs: Input arrays to be flattened.
        """
        outputs = super().__call__(inputs)

        hooks.add_summary(None, self.__class__.__name__, outputs)

        return outputs
