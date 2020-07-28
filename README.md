# Elegy

[![PyPI Status Badge](https://badge.fury.io/py/elegy.svg)](https://pypi.org/project/elegy/)
[![Coverage](https://img.shields.io/codecov/c/github/poets-ai/elegy?color=%2334D058)](https://codecov.io/gh/poets-ai/elegy)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/elegy)](https://pypi.org/project/elegy/)
[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://poets-ai.github.io/elegy/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/poets-ai/elegy/issues)
[![Status](https://github.com/poets-ai/elegy/workflows/GitHub%20CI/badge.svg)](https://github.com/poets-ai/elegy/actions?query=workflow%3A"GitHub+CI")

-----------------

_Elegy is a Neural Networks framework based on Jax and Haiku._ 

Elegy implements the Keras API but makes changes to play better with Jax & Haiku and gives more flexibility around losses and metrics (more on this soon). Elegy is still in a very early stage, feel free to test it and send us your feedback!

#### Main Features

* **Familiar**: Elegy should feel very familiar to Keras users.
* **Flexible**: Elegy improves upon the basic Keras API by letting users optionally take more control over the definition of losses and metrics.
* **Easy-to-use**: Elegy maintains all the simplicity and ease of use that Keras brings with it.
* **Compatible**: Elegy strives to be compatible with the rest of the Jax and Haiku ecosystem.

For more information take a look at the [Documentation](https://poets-ai.github.io/elegy).

## Installation

Install Elegy using pip:
```bash
pip install elegy
```

For Windows users we recommend the Windows subsystem for linux 2 [WSL2](https://docs.microsoft.com/es-es/windows/wsl/install-win10?redirectedfrom=MSDN) since [jax](https://github.com/google/jax/issues/438) does not support it yet.

## Quick Start
Elegy greatly simplifies the training of Deep Learning models compared to pure Jax / Haiku where, due to Jax functional nature, users have to do a lot of book keeping around the state of the model. In Elegy you just have to follow 3 basic steps:

**1.** Define the architecture inside an `elegy.Module`:
```python
class MLP(elegy.Module):
    def __apply__(self, image: jnp.ndarray) -> jnp.ndarray:
        mlp = hk.Sequential([
            hk.Flatten(),
            hk.Linear(300),
            jax.nn.relu,
            hk.Linear(10),
        ])
        return mlp(image)
```
**2.** Create a `Model` from this module and specify additional things like losses, metrics, and optimizers:
```python
model = elegy.Model(
    module=MLP.defer(),
    loss=[
        elegy.losses.SparseCategoricalCrossentropy(from_logits=True),
        elegy.regularizers.GlobalL2(l=1e-5),
    ],
    metrics=elegy.metrics.SparseCategoricalAccuracy.defer(),
    optimizer=optix.rmsprop(1e-3),
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
* Couple of examples in [examples](https://github.com/poets-ai/elegy/tree/master/examples) directory.
* Haiku's [User Manual](https://github.com/deepmind/dm-haiku#user-manual) and [Documentation](https://dm-haiku.readthedocs.io/en/latest/)
* [What is Jax?](https://github.com/google/jax#what-is-jax)

## Why Jax + Haiku?

**Jax** is a linear algebra library with the perfect recipe:
* Numpy's familiar API
* The speed and hardware support of XLA
* Automatic Differentiation

The awesome thing about Jax is that Deep Learning is just a usecase that it happens to excel at but you can use it for most task you would use NumPy for.

On the other hand, **Haiku** is a Neural Networks library built on top of Jax that implements a `Module` system, common Neural Network layers, and even some full architectures. Compared to other Jax-based libraries like Trax or Flax, Haiku is very minimal, polished, well documented, and makes it super easy / clean to implement Deep Learning code! 

We believe that **Elegy** can offer the best experience for coding Deep Learning applications by leveraging the power and familiarity of Jax API, the ease-of-use of Haiku's Module system, and packaging everything on top of a convenient Keras-like API.

## Features
* `Model` estimator class
* `losses` module
* `metrics` module
* `regularizers` module
* `callbacks` module
* `nn` layers module

For more information checkout the **Reference API** section in the [Documentation](https://poets-ai.github.io/elegy).

## Contributing
Deep Learning is evolving at an incredible pace, there is so much to do and so few hands. If you wish to contibute anything from a loss or metric to a new awesome feature for Elegy just open an issue or send a PR! For more information check out our [Contibuting Guide](https://poets-ai.github.io/elegy/guides/contributing).

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
title = {Elegy: A Keras-like deep learning framework based on Jax & Haiku},
url = {https://github.com/poets-ai/elegy},
version = {0.1.5},
year = {2020},
}
```

Where the current *version* may be retrieved either from the `Release` tag or the file [elegy/\_\_init\_\_.py](https://github.com/poets-ai/elegy/blob/master/elegy/__init__.py) and the *year* corresponds to the project's release year.