# Contributing
This is a short guide on how to start contributing to Elegy along with some best practices for the project.

## Setup
We use `poetry >= 1.1.4`, the easiest way to setup a development environment is run:

```bash
poetry config virtualenvs.in-project true --local
poetry install
```

In order for Jax to recognize your GPU, you will probably have to install it again using the command below.

```bash
PYTHON_VERSION=cp38  
CUDA_VERSION=cuda101  # alternatives: cuda100, cuda101, cuda102, cuda110, check your cuda version
PLATFORM=manylinux2010_x86_64  # alternatives: manylinux2010_x86_64
BASE_URL='https://storage.googleapis.com/jax-releases'
pip install --upgrade $BASE_URL/$CUDA_VERSION/jaxlib-0.1.55-$PYTHON_VERSION-none-$PLATFORM.whl
pip install --upgrade jax  
```

#### Gitpod
An alternative way to contribute is using [gitpod](https://gitpod.io/) which creates a vscode-based cloud development enviroment.
To get started just login at gitpod, grant the appropriate permissions to github, and open the following link:

https://gitpod.io/#https://github.com/poets-ai/elegy

We have built a `python 3.8` enviroment and all development dependencies will install when the enviroment starts.

## Creating Losses and Metrics
For this you can follow these guidelines:

* Each loss / metric should be defined in its own file.
* Inherit from either `elegy.losses.loss.Loss` or `elegy.metrics.metric.Metric` or an existing class that inherits from them.
* Try to use an existing metric or loss as a template
* You must provide documentation for the following:
    * The class definition.
    * The `__init__` method.
    * The `call` method.
* Try to port the documentation + signature from its Keras counter part.
    * If so you must give credits to the original source file.
* You must include tests.
    * If you there exists an equivalent loss/metric in Keras you must test numerical equivalence between both.

## Testing
To execute all the tests just run
```bash
pytest
```

## Documentation
We use `mkdocs`. If you create a new object that requires documentation please do the following:

* Add a markdown file inside `/docs/api` in the appropriate location according to the project's structure. This file must:
    * Contain the path of function / class as header
    * Use `mkdocstring` to render the API information.
    * Example:
```markdown
# elegy.losses.BinaryCrossentropy

::: elegy.losses.BinaryCrossentropy
    selection:
        inherited_members: true
        members:
            - call
            - __init__
```
* Add and entry to `mkdocs.yml` inside `nav` pointing to this file. Checkout `mkdocs.yml`.

To build and visualize the documentation locally run
```bash
mkdocs serve
```

## Creating a PR
Before sending a pull request make sure all test run and code is formatted with `black`:

```bash
black .
```

## Changelog
`CHANGELOG.md` is automatically generated using [github-changelog-generator](https://github.com/github-changelog-generator/github-changelog-generator), to update the changelog just run:
```bash
docker run -it --rm -v (pwd):/usr/local/src/your-app ferrarimarco/github-changelog-generator -u poets-ai -p elegy -t <TOKEN>
```
where `<TOKEN>` token can be obtained from Github at [Personal access tokens](https://github.com/settings/tokens), you only have to give permission for the `repo` section.
