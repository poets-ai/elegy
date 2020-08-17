# Contributing
This is a short guide on how to start contibuting to Elegy along with some best practices for the project.

## Setup
We use `poetry` so the easiest way to setup a development environment is run

```bash
poetry install
```

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

# Creating a PR
Before sending a pull request make sure all test run and code is formatted with `black`:

```bash
black .
```