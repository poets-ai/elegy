# Elegy

[![PyPI Status Badge](https://badge.fury.io/py/elegy.svg)](https://pypi.org/project/elegy/)
[![Coverage](https://img.shields.io/codecov/c/github/poets-ai/elegy?color=%2334D058)](https://codecov.io/gh/poets-ai/elegy)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/elegy)](https://pypi.org/project/elegy/)
[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://poets-ai.github.io/elegy/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/poets-ai/elegy/issues)
[![Status](https://github.com/poets-ai/elegy/workflows/GitHub%20CI/badge.svg)](https://github.com/poets-ai/elegy/actions?query=workflow%3A"GitHub+CI")

-----------------

_Elegy is a Neural Networks framework based on Jax inspired by Keras._  

Elegy implements the Keras API but makes changes to play better with Jax and gives more flexibility around [losses and metrics](https://poets-ai.github.io/elegy/guides/modules-losses-metrics/) and excellent [module system](https://poets-ai.github.io/elegy/guides/module-system/) that makes it super easy to use. Elegy is in an early stage, feel free to send us your feedback!

#### Main Features

* **Familiar**: Elegy should feel very familiar to Keras users.
* **Flexible**: Elegy improves upon the basic Keras API by letting users optionally take more control over the definition of losses and metrics.
* **Easy-to-use**: Elegy maintains all the simplicity and ease of use that Keras brings with it.
* **Compatible**: Elegy strives to be compatible with the rest of the Jax ecosystem.

For more information take a look at the [Documentation](https://poets-ai.github.io/elegy).

## Installation

Install Elegy using pip:
```bash
pip install elegy
```

For Windows users we recommend the Windows subsystem for linux 2 [WSL2](https://docs.microsoft.com/es-es/windows/wsl/install-win10?redirectedfrom=MSDN) since [jax](https://github.com/google/jax/issues/438) does not support it yet.

## Quick Start
Elegy greatly simplifies the training of Deep Learning models compared to pure Jax where, due to Jax's functional nature, users have to do a lot of book keeping around the state of the model. In Elegy you just have to follow 3 basic steps:

**1.** Define the architecture inside an `elegy.Module`:
```python
class MLP(elegy.Module):
    def call(self, x: jnp.ndarray) -> jnp.ndarray:
        x = elegy.nn.Linear(300)(x)
        x = jax.nn.relu(x)
        x = elegy.nn.Linear(10)(x)
        return x
```
Note that we can define sub-modules on-the-fly directly in the `call` (forward) method.

**2.** Create a `Model` from this module and specify additional things like losses, metrics, and optimizers:
```python
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

And you are done! For more information check out:


* Our [Getting Started](https://poets-ai.github.io/elegy/getting-started/) tutorial.
* Elegy's [Documentation](https://poets-ai.github.io/elegy).
* The [examples](https://github.com/poets-ai/elegy/tree/master/examples) directory.
* [What is Jax?](https://github.com/google/jax#what-is-jax)

## Why Jax & Elegy?

Given all the well-stablished Deep Learning framework like TensorFlow + Keras or Pytorch + Pytorch-Lightning/Skorch, it is fair to ask why we need something like Jax + Elegy? Here are some of the reasons why this framework exists.

#### Why Jax?

**Jax** is a linear algebra library with the perfect recipe:
* Numpy's familiar API
* The speed and hardware support of XLA
* Automatic Differentiation

The awesome thing about Jax is that Deep Learning is just a use-case that it happens to excel at but you can use it for most task you would use NumPy for. Jax is so compatible with Numpy that is array type actually inherits from `np.ndarray`.

In a sense, Jax takes the best of both TensorFlow and Pytorch in a principled manner: while both TF and Pytorch historically converged to the same set of features, their APIs still contain quirks they have to keep for compatibility.

#### Why Elegy?

We believe that **Elegy** can offer the best experience for coding Deep Learning applications by leveraging the power and familiarity of Jax API, an easy-to-use and succinct Module system, and packaging everything on top of a convenient Keras-like API. Elegy improves upon other Deep Learning frameworks in the following ways:

1. Its hook-based [Module System](https://poets-ai.github.io/elegy/guides/module-system/) makes it easier (less verbose) to write model code compared to Keras & Pytorch since it lets you declare sub-modules, parameters, and states directly on your `call` (forward) method. Thanks to this you get shape inference for free so there is no need for a `build` method (Keras) or propagating shape information all over the place (Pytorch). A naive implementation of `Linear` could be as simple as:

```python
class Linear(elegy.Module):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def call(self, x):
        w = self.add_parameter("w", [x.shape[-1], self.units], initializer=jnp.ones)
        b = self.add_parameter("b", [self.units], initializer=jnp.ones)

        return jnp.dot(x, w) + b
```
2. It has a very flexible system for defining the inputs for [losses and metrics](https://poets-ai.github.io/elegy/guides/modules-losses-metrics/) based on _dependency injection_ in opposition to Keras rigid requirement to have matching (output, label) pairs, and being unable to use additional information like inputs, parameters, and states in the definition of losses and metrics. 
3. Its hook system preserve's [reference information](https://poets-ai.github.io/elegy/guides/module-system/) from a module to its sub-modules, parameters, and states while maintaining a functional API. This is crucial since most Jax-based frameworks like Flax and Haiku tend to loose this information which makes it very tricky to perform tasks like transfer learning where you need to mix a pre-trained models into a new model (easier to do if you keep references).

## Features
* `Model` estimator class
* `losses` module
* `metrics` module
* `regularizers` module
* `callbacks` module
* `nn` layers module

For more information checkout the **Reference API** section in the [Documentation](https://poets-ai.github.io/elegy).

## Contributing
Deep Learning is evolving at an incredible pace, there is so much to do and so few hands. If you wish to contibute anything from a loss or metric to a new awesome feature for Elegy just open an issue or send a PR! For more information check out our [Contributing Guide](https://poets-ai.github.io/elegy/guides/contributing).

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
title = {Elegy: A Keras-like deep learning framework based on Jax},
url = {https://github.com/poets-ai/elegy},
version = {0.3.0},
year = {2020},
}
```


Where the current *version* may be retrieved either from the `Release` tag or the file [elegy/\_\_init\_\_.py](https://github.com/poets-ai/elegy/blob/master/elegy/__init__.py) and the *year* corresponds to the project's release year.