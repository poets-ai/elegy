# Elegy

<!-- [![PyPI Status Badge](https://badge.fury.io/py/eg.svg)](https://pypi.org/project/elegy/) -->
<!-- [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/elegy)](https://pypi.org/project/elegy/) -->
<!-- [![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://poets-ai.github.io/elegy/) -->
<!-- [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) -->
[![Coverage](https://img.shields.io/codecov/c/github/poets-ai/elegy?color=%2334D058)](https://codecov.io/gh/poets-ai/elegy)
[![Status](https://github.com/poets-ai/elegy/workflows/GitHub%20CI/badge.svg)](https://github.com/poets-ai/elegy/actions?query=workflow%3A%22GitHub+CI%22)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/poets-ai/elegy/issues)

______________________________________________________________________

_A High Level API for Deep Learning in JAX_

#### Main Features

- 😀 **Easy-to-use**: Elegy provides a Keras-like high-level API that makes it very easy to do common tasks.
- 💪‍ **Flexible**: Elegy provides a Pytorch Lightning-like low-level API that offers maximum flexibility when needed.
- 🔌 **Compatible**: Elegy can consume many familiar data sources, including TensorFlow Datasets, Pytorch DataLoaders, Python generators, and Array Pytrees.
<!-- - **Agnostic**: Elegy supports various frameworks, including Flax, Haiku, and Optax on the high-level API, and it is 100% framework-agnostic on the low-level API. -->

Elegy is built on top of [Treex](https://github.com/cgarciae/treex) and [Treeo](https://github.com/cgarciae/treeo) and reexports their APIs for convenience. 


[Getting Started](https://poets-ai.github.io/elegy/getting-started/high-level-api) | <!--[User Guide](https://cgarciae.github.io/treex/user-guide/intro)--> | [Examples](#examples) | [Documentation](https://poets-ai.github.io/elegy)


## What is included?
* A `Model` class with an Estimator-like API.
* A `callbacks` module with common Keras callbacks.

##### From Treex
* A `Module` class.
* A `nn` module for with common layers.
* A `losses` module with common loss functions.
* A `metrics` module with common metrics.

## Installation

Install using pip:

```bash
pip install elegy
```

For Windows users, we recommend the Windows subsystem for Linux 2 [WSL2](https://docs.microsoft.com/es-es/windows/wsl/install-win10?redirectedfrom=MSDN) since [jax](https://github.com/google/jax/issues/438) does not support it yet.

## Quick Start: High-level API

Elegy's high-level API provides a straightforward interface you can use by implementing the following steps:

**1.** Define the architecture inside a `Module`:

```python
import jax
import elegy as eg

class MLP(eg.Module):
    @eg.compact
    def __call__(self, x):
        x = eg.Linear(300)(x)
        x = jax.nn.relu(x)
        x = eg.Linear(10)(x)
        return x
```

**2.** Create a `Model` from this module and specify additional things like losses, metrics, and optimizers:

```python
import optax optax
import elegy as eg

model = eg.Model(
    module=MLP(),
    loss=[
        eg.losses.Crossentropy(),
        eg.regularizers.L2(l=1e-5),
    ],
    metrics=eg.metrics.Accuracy(),
    optimizer=optax.rmsprop(1e-3),
)
```

**3.** Train the model using the `fit` method:

```python
model.fit(
    x=X_train,
    y=y_train,
    epochs=100,
    steps_per_epoch=200,
    batch_size=64,
    validation_data=(X_test, y_test),
    shuffle=True,
    callbacks=[eg.callbacks.TensorBoard("summaries")]
)
```

## Quick Start: Low-level API

Elegy's low-level API lets you explicitly define what goes on during training, testing, and inference. Let's define our own custom `Model` to implement a `LinearClassifier` with pure JAX:

**1.** Define a custom `init_step` method:

```python
class LinearClassifier(eg.Model):
    # use treex's API to declare parameter nodes
    w: jnp.ndarray = eg.Parameter.node()
    b: jnp.ndarray = eg.Parameter.node()

    def init_step(self, key: jnp.ndarray, inputs: jnp.ndarray):
        self.w = jax.random.uniform(
            key=key,
            shape=[features_in, 10],
        )
        self.b = jnp.zeros([10])

        self.optimizer = self.optimizer.init(self)

        return self
```
Here we declared the parameters `w` and `b` using Treex's `Parameter.node()` for pedagogical reasons, however normally you don't have to do this since you typically use a sub-`Module` instead.

**2.** Define a custom `test_step` method:
```python
    def test_step(
        self,
        inputs,
        labels,
    ):
        # flatten + scale
        inputs = jnp.reshape(inputs, (inputs.shape[0], -1)) / 255

        # forward
        logits = jnp.dot(inputs, self.w) + self.b

        # crossentropy loss
        target = jax.nn.one_hot(labels["target"], 10)
        loss = optax.softmax_cross_entropy(logits, target).mean()

        # metrics
        logs = dict(
            acc=jnp.mean(jnp.argmax(logits, axis=-1) == labels["target"]),
            loss=loss,
        )

        return loss, logs, self
```

**3.** Instantiate our `LinearClassifier` with an optimizer:

```python
model = LinearClassifier(
    optimizer=optax.rmsprop(1e-3),
)
```

**4.** Train the model using the `fit` method:

```python
model.fit(
    x=X_train,
    y=y_train,
    epochs=100,
    steps_per_epoch=200,
    batch_size=64,
    validation_data=(X_test, y_test),
    shuffle=True,
    callbacks=[eg.callbacks.TensorBoard("summaries")]
)
```

#### Using Jax Frameworks

It is straightforward to integrate other functional JAX libraries with this
low-level API, here is an example with Flax:

```python
class LinearClassifier(eg.Model):
    params: Mapping[str, Any] = eg.Parameter.node()
    batch_stats: Mapping[str, Any] = eg.BatchStat.node()
    next_key: eg.KeySeq

    def init_step(self, key, inputs):
        self.next_key = eg.KeySeq(key)

        _, variables = self.module.init_with_output(
            {"params": self.next_key(), "dropout": self.next_key()}, x
        )
        self.params = variables["params"]
        self.batch_stats = variables["batch_stats"]

        self.optimizer = self.optimizer.init(self.parameters())

    def test_step(self, inputs, labels):
        # forward
        variables = dict(
            params=self.params,
            batch_stats=self.batch_stats,
        )
        logits, variables = self.module.apply(
            variables,
            inputs, 
            rngs={"dropout": self.next_key()}, 
            mutable=True,
        )
        self.batch_stats = variables["batch_stats"]
        
        # loss
        target = jax.nn.one_hot(labels["target"], 10)
        loss = optax.softmax_cross_entropy(logits, target).mean()

        # logs
        logs = dict(
            accuracy=accuracy,
            loss=loss,
        )
        return loss, logs, self
```
Here `module` is a `flax.linen.Module` 


### Examples

To run the examples, first install some required packages:

```bash
pip install -r examples/requirements.txt
```

Now run the example:

```bash
python examples/flax_mnist_vae.py
```

## Contributing

Deep Learning is evolving at an incredible pace, and there is so much to do and so few hands. If you wish to contribute anything from a loss or metric to a new awesome feature for Elegy, open an issue or send a PR! For more information, check out our [Contributing Guide](https://poets-ai.github.io/elegy/guides/contributing).

## About Us

We are some friends passionate about ML.

## License

This project is [licensed under the Apache v2.0 License](LICENSE).

## Citing Elegy

To cite this project:

**BibTeX**

```
@software{elegy2020repository,
author = {PoetsAI},
title = {Elegy: A High Level API for Deep Learning in JAX},
url = {https://github.com/poets-ai/elegy},
version = {0.7.4},
year = {2021},
}
```

The current *version* may be retrieved either from the `Release` tag or the file [elegy/\_\_init\_\_.py](https://github.com/poets-ai/elegy/blob/master/elegy/__init__.py) and the *year* corresponds to the project's release year.
