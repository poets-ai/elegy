import typing as tp
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from elegy import utils


class Regularizer(ABC):
    """Regularizer base class.

    Regularizers allow you to apply penalties on layer parameters or layer
    activity during optimization. These penalties are summed into the loss
    function that the network optimizes.

    Regularization penalties are applied on a per-layer basis. The exact API will
    depend on the layer, but many layers (e.g. `Dense`, `Conv1D`, `Conv2D` and
    `Conv3D`) have a unified API.

    These layers expose 3 keyword arguments:

    - `kernel_regularizer`: Regularizer to apply a penalty on the layer's kernel
    - `bias_regularizer`: Regularizer to apply a penalty on the layer's bias
    - `activity_regularizer`: Regularizer to apply a penalty on the layer's output

    All layers (including custom layers) expose `activity_regularizer` as a
    settable property, whether or not it is in the constructor arguments.

    The value returned by the `activity_regularizer` is divided by the input
    batch size so that the relative weighting between the weight regularizers and
    the activity regularizers does not change with the batch size.

    You can access a layer's regularization penalties by calling `layer.losses`
    after calling the layer on inputs.

    ## Example

    >>> layer = tf.keras.layers.Dense(
    ...     5, input_dim=5,
    ...     kernel_initializer='ones',
    ...     kernel_regularizer=tf.keras.regularizers.L1(0.01),
    ...     activity_regularizer=tf.keras.regularizers.L2(0.01))
    >>> tensor = tf.ones(shape=(5, 5)) * 2.0
    >>> out = layer(tensor)

    >>> # The kernel regularization term is 0.25
    >>> # The activity regularization term (after dividing by the batch size) is 5
    >>> tf.math.reduce_sum(layer.losses)
    <tf.Tensor: shape=(), dtype=float32, numpy=5.25>

    ## Available penalties

    ```python
    tf.keras.regularizers.L1(0.3)  # L1 Regularization Penalty
    tf.keras.regularizers.L2(0.1)  # L2 Regularization Penalty
    tf.keras.regularizers.L1L2(l1=0.01, l2=0.01)  # L1 + L2 penalties
    ```

    ## Directly calling a regularizer

    Compute a regularization loss on a tensor by directly calling a regularizer
    as if it is a one-argument function.

    E.g.
    >>> regularizer = tf.keras.regularizers.L2(2.)
    >>> tensor = tf.ones(shape=(5, 5))
    >>> regularizer(tensor)
    <tf.Tensor: shape=(), dtype=float32, numpy=50.0>


    ## Developing new regularizers

    Any function that takes in a weight matrix and returns a scalar
    tensor can be used as a regularizer, e.g.:

    >>> @tf.keras.utils.register_keras_serializable(package='Custom', name='l1')
    ... def l1_reg(weight_matrix):
    ...    return 0.01 * tf.math.reduce_sum(tf.math.abs(weight_matrix))
    ...
    >>> layer = tf.keras.layers.Dense(5, input_dim=5,
    ...     kernel_initializer='ones', kernel_regularizer=l1_reg)
    >>> tensor = tf.ones(shape=(5, 5))
    >>> out = layer(tensor)
    >>> layer.losses
    [<tf.Tensor: shape=(), dtype=float32, numpy=0.25>]

    Alternatively, you can write your custom regularizers in an
    object-oriented way by extending this regularizer base class, e.g.:

    >>> @tf.keras.utils.register_keras_serializable(package='Custom', name='l2')
    ... class L2Regularizer(tf.keras.regularizers.Regularizer):
    ...   def __init__(self, l2=0.):  # pylint: disable=redefined-outer-name
    ...     self.l2 = l2
    ...
    ...   def __call__(self, x):
    ...     return self.l2 * tf.math.reduce_sum(tf.math.square(x))
    ...
    ...   def get_config(self):
    ...     return {'l2': float(self.l2)}
    ...
    >>> layer = tf.keras.layers.Dense(
    ...   5, input_dim=5, kernel_initializer='ones',
    ...   kernel_regularizer=L2Regularizer(l2=0.5))

    >>> tensor = tf.ones(shape=(5, 5))
    >>> out = layer(tensor)
    >>> layer.losses
    [<tf.Tensor: shape=(), dtype=float32, numpy=12.5>]

    ### A note on serialization and deserialization:

    Registering the regularizers as serializable is optional if you are just
    training and executing models, exporting to and from SavedModels, or saving
    and loading weight checkpoints.

    Registration is required for Keras `model_to_estimator`, saving and
    loading models to HDF5 formats, Keras model cloning, some visualization
    utilities, and exporting models to and from JSON. If using this functionality,
    you must make sure any python process running your model has also defined
    and registered your custom regularizer.

    `tf.keras.utils.register_keras_serializable` is only available in TF 2.1 and
    beyond. In earlier versions of TensorFlow you must pass your custom
    regularizer to the `custom_objects` argument of methods that expect custom
    regularizers to be registered as serializable.
    """

    def __init__(self, name: tp.Optional[str] = None):
        self.name = name if name is not None else utils.get_name(self)

    def __call__(self, parameters: tp.Dict) -> jnp.ndarray:
        return sum((self.call(p) for p in jax.tree_leaves(parameters)), jnp.array(0.0))

    @abstractmethod
    def call(self, *args, **kwargs) -> tp.Any:
        ...
