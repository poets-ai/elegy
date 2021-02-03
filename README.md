# Elegy
[![PyPI Status Badge](https://badge.fury.io/py/elegy.svg)](https://pypi.org/project/elegy/)
[![Coverage](https://img.shields.io/codecov/c/github/poets-ai/elegy?color=%2334D058)](https://codecov.io/gh/poets-ai/elegy)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/elegy)](https://pypi.org/project/elegy/)
[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://poets-ai.github.io/elegy/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/poets-ai/elegy/issues)
[![Status](https://github.com/poets-ai/elegy/workflows/GitHub%20CI/badge.svg)](https://github.com/poets-ai/elegy/actions?query=workflow%3A"GitHub+CI")

-----------------

_Elegy is a framework-agnostic Trainer interface for the Jax ecosystem._  

#### Main Features
* **Easy-to-use**: Elegy provides a Keras-like high-level API that makes it very easy to do common tasks.
* **Flexible**: Elegy provides a functional Pytorch Lightning-like low-level API that provides maximal flexibility when needed.
* **Agnostic**: Elegy provides support a variety of frameworks including Flax, Haiku, and Optax on the high-level API, and it is 100% framework-agnostic on the low-level API.
* **Compatible**: Elegy can consume a wide variety of common data sources including TensorFlow Datasets, Pytorch DataLoaders, Python generators, and Numpy pytrees.

For more information take a look at the [Documentation](https://poets-ai.github.io/elegy).

## Installation

Install Elegy using pip:
```bash
pip install elegy
```

For Windows users we recommend the Windows subsystem for linux 2 [WSL2](https://docs.microsoft.com/es-es/windows/wsl/install-win10?redirectedfrom=MSDN) since [jax](https://github.com/google/jax/issues/438) does not support it yet.

## Quick Start: High-level API
Elegy's high-level API provides a very simple interface you can use by implementing following steps:

**1.** Define the architecture inside a `Module`. We will use Flax Linen for this example:
```python
import flax.linen as nn
import jax

class MLP(nn.Module):
    @nn.compact
    def call(self, x):
        x = nn.Dense(300)(x)
        x = jax.nn.relu(x)
        x = nn.Dense(10)(x)
        return x
```

**2.** Create a `Model` from this module and specify additional things like losses, metrics, and optimizers:
```python
import elegy, optax

model = elegy.Model(
    module=MLP(),
    loss=[
        elegy.losses.SparseCategoricalCrossentropy(from_logits=True),
        elegy.regularizers.GlobalL2(l=1e-5),
    ],
    metrics=elegy.metrics.SparseCategoricalAccuracy(),
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
    callbacks=[elegy.callbacks.TensorBoard("summaries")]
)
```

## Quick Start: Low-level API
In Elegy's low-level API lets you define exactly what goes on during training, testing, and inference. Lets define the `test_step` to implement a linear classifier in pure jax:

**1.** Calculate our loss, logs, and states:
```python
class LinearClassifier(elegy.Model):
    # request parameters by name via depending injection.
    # possible: net_params, x, y_true, net_states, metrics_states, sample_weight, class_weight, rng, states, initializing
    def test_step(
        self,
        x, # inputs
        y_true, # labels
        states: elegy.States, # model state
        initializing: bool, # if True we should initialize our parameters
    ):  
        # flatten + scale
        x = jnp.reshape(x, (x.shape[0], -1)) / 255
        # initialize or use existing parameters
        if initializing:
            w = jax.random.uniform(
                jax.random.PRNGKey(42), shape=[np.prod(x.shape[1:]), 10]
            )
            b = jax.random.uniform(jax.random.PRNGKey(69), shape=[1])
        else:
            w, b = states.net_params
        # model
        logits = jnp.dot(x, w) + b
        # categorical crossentropy loss
        labels = jax.nn.one_hot(y_true, 10)
        loss = jnp.mean(-jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1))
        accuracy=jnp.mean(jnp.argmax(logits, axis=-1) == y_true)
        # metrics
        logs = dict(
            accuracy=accuracy,
            loss=loss,
        )
        return loss, logs, states.update(rng=rng, net_params=(w, b))
```

**2.** Instantiate our `LinearClassifier` with an optimizer:
```python
model = LinearClassifier(
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
    callbacks=[elegy.callbacks.TensorBoard("summaries")]
)
```
#### Using Jax Frameworks
It is straightforward to integrate other functional JAX libraries with this 
low-level API:

```python
class LinearClassifier(elegy.Model):
    def test_step(
        self, x, y_true, states: elegy.States, initializing: bool, rng: elegy.RNGSeq
    ):
        x = jnp.reshape(x, (x.shape[0], -1)) / 255
        if initializing:
            logits, variables = self.module.init_with_output(
                {"params": rng.next(), "dropout": rng.next()}, x
            )
        else:
            variables = dict(params=states.net_params, **states.net_states)
            logits, variables = self.module.apply(
                variables, x, rngs={"dropout": rng.next()}, mutable=True
            )
        net_states, net_params = variables.pop("params")
        
        labels = jax.nn.one_hot(y_true, 10)
        loss = jnp.mean(-jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1))
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y_true)

        logs = dict(accuracy=accuracy, loss=loss)
        return loss, logs, states.update(rng=rng, net_params=net_params, net_states=net_states)
```

## More Info
* [Getting Started: High-level API](https://poets-ai.github.io/elegy/getting-started-high-level-api/) tutorial.
* [Getting Started: Low-level API](https://poets-ai.github.io/elegy/getting-started-low-level-api/) tutorial.
* Elegy's [Documentation](https://poets-ai.github.io/elegy).
* The [examples](https://github.com/poets-ai/elegy/tree/master/examples) directory.
* [What is Jax?](https://github.com/google/jax#what-is-jax)

## Contributing
Deep Learning is evolving at an incredible pace, there is so much to do and so few hands. If you wish to contribute anything from a loss or metric to a new awesome feature for Elegy just open an issue or send a PR! For more information check out our [Contributing Guide](https://poets-ai.github.io/elegy/guides/contributing).

## About Us
We are some friends passionate about ML.

## License
Apache

## Citing Elegy

To cite this project:

**BibTeX**

```
@software{elegy2020repository,
author = {PoetsAI},
title = {Elegy: A framework-agnostic Trainer interface for the Jax ecosystem},
url = {https://github.com/poets-ai/elegy},
version = {0.4.1},
year = {2020},
}
```


Where the current *version* may be retrieved either from the `Release` tag or the file [elegy/\_\_init\_\_.py](https://github.com/poets-ai/elegy/blob/master/elegy/__init__.py) and the *year* corresponds to the project's release year.
