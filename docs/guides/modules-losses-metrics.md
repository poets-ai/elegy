# Modules, Losses, and Metrics

This guide goes into depth on how modules, losses and metrics work in Elegy when used with an `elegy.Model`. For more in-depth explanation on how they work internally check out the [Module System](../module-system) guide.

## Keras Limitations
One of our goals with Elegy was to solve Keras restrictions around the type of losses and metrics you can define. When creating a complex model with multiple outputs in Keras, say `output_a` and `output_b`, you are forced to define losses and metrics per-output only:

```python
model.compile(
    loss={
        "output_a": keras.losses.BinaryCrossentropy(from_logits=True),
        "output_b": keras.losses.CategoricalCrossentropy(from_logits=True),
    },
    metrics={
        "output_a": keras.losses.BinaryAccuracy(from_logits=True),
        "output_b": keras.losses.CategoricalAccuracy(from_logits=True),
    },
    ...
)
```
This is very restrictive, in particular it doesn't allow the following:

1. Losses and metrics that combine multiple outputs with multiple labels.
2. A single loss/metric based on multiple outputs (a especial case of the previous).
3. Losses and metrics that depend on other variables such as inputs, parameters, states, etc.

Most of these are usually solvable by tricks such as:

* [Concatenating the outputs / labels](https://stackoverflow.com/a/57030727/2118130)
* Passing the inputs and other kind of information as labels.
* Using the functional API which is more flexible (but it only runs on graph mode making it very painful to debug).
 
It is clear that these solution are hacky. Sometimes they are non-obvious, and depending on the problem they can be insufficient.

## Dependency Injection
Elegy solves the previous problems by introducing a _dependency injection_ mechanism that allows the user to express complex functions by simply declaring the variables to use **by their name**. The following parameters are available for the different callables you pass to Elegy:


 | parameter       | Description                                                    | Module | Metric | Loss |
 | --------------- | -------------------------------------------------------------- | ------ | ------ | ---- |
 | `x`             | Inputs of the model corresponding to the `x` argument of `fit` | *      | x      | x    |
 | `y_true`        | The input labels corresponding to the `y` argument of `fit`    |        | x      | x    |
 | `y_pred`        | Outputs of the model                                           |        | x      | x    |
 | `sample_weight` | Importance of each sample                                      |        | x      | x    |
 | `class_weight`  | Importance of each class                                       |        | x      | x    |
 | `training`      | Whether training is currently in progress                      | x      | x      | x    |
 | `parameters`    | The learnable parameters of the model                          |        | x      | x    |
 | `states`        | The non-learnable parameters of the model                      |        | x      | x    |


!!! Note
    The content of `x` is technically passed to the model's `Module` but the parameter name **"x"** will bare no special meaning in that context.


## Modules
Modules define the architecture of the network, their primary task (in Elegy terms) is transforming the inputs `x` into outputs `y_pred`. To make it easy to consume the content of `x`, Elegy has some special but very simple rules on how the signature of any `Module` can be structured:

**1.** If `x` is simply an array it will be passed directly:

```python hl_lines="2 10"
class SomeModule:
    def call(self, m):
        ...

model = elegy.Model(SomeModule(), ...)

a = get_inputs()

model.fit(
    x=a,
    ...
)
```
In this case `a` is passed as `m`.

**2.** If `x` is a `tuple`, then `x` will be expanded positional arguments a.k.a. `*args`, this means that the module will have define **exactly** as many arguments as there are inputs. For example:
  
```python hl_lines="2 10"
class SomeModule:
    def call(self, m, n):
        ...

model = elegy.Model(SomeModule(), ...)

a, b = get_inputs()

model.fit(
    x=(a, b),
    ...
)
```
In this case `a` is passed as `m` and `b` is passed as `n`.

**3.** If `x` is a `dict`, then `x` will be expanded as keyword arguments a.k.a. `**kwargs`, in this case the module can optionally request any key defined in `x` as an argument. For example:

```python hl_lines="2 10"
class SomeModule:
    def call(self, n, o):
        ...

model = elegy.Model(SomeModule(), ...)

a, b, c = get_inputs()

model.fit(
    x={"m": a, "n": b, "o": c},
    ...
)
```
Here only `n` and `o` are requested by name, and you get as input its values `b` and `c`, the variable `m` with the content of `a` is safely ignored. If you want to request all the available inputs you can use `**kwargs`.

## Losses
Losses can request all the available parameters that Elegy provides for dependency injection. A typical loss will request the `y_true` and `y_pred` values (as its common / enforced in Keras). The Mean Squared Error loss for example is easily defined in these terms:

```python hl_lines="2"
class MSE(elegy.Loss): 
    def call(self, y_true, y_pred):
        return jnp.mean(jnp.square(y_true - y_pred), axis=-1)

...

X_train, y_train = get_inputs()

model.fit(
    x=X_train,
    y=y_train,
    loss=MSE(),
)
```
Here the input `y` is passed as `y_true` to `MSE`. For an auto-encoder however, it makes perfect sense to define the loss only in terms of `x` (according to the math) and Elegy lets you do exactly that:

```python hl_lines="2"
class AutoEncoderLoss(elegy.Loss): 
    def call(self, x, y_pred):
        return jnp.mean(jnp.square(x - y_pred), axis=-1)

...

X_train, _ = get_inputs()

model.fit(
    x=X_train,
    loss=AutoEncoderLoss(),
)
```

Notice thanks to this we didn't have to define `y` on the `fit` method.

!!! Note
    An alternative here is to just use the previous definition of `MSE` and define `y=X_train`. However, avoiding the creation of redundant information is good in general and being explicit about dependencies helps documenting the behaviour of the model.

### Partitioning a loss
If you have a complex loss function that is just a sum of different parts that have to be compute together you might define something like this:
```python
class SomeComplexFunction(elegy.Loss): 
    def call(self, x, y_true, y_pred, parameters, ...):
        ...
        return a + b + c
```
Elegy lets you return a `dict` specifying the name of each part:

```python
class SomeComplexFunction(elegy.Loss): 
    def call(self, x, y_true, y_pred, parameters, ...):
        ...
        return {
            "a": a,
            "b": b,
            "c": c,
        }
```
Elegy will use this information to show you each loss separate in the logs / Tensorboard / History with the names:

* `some_complex_function_loss/a`
* `some_complex_function_loss/b`
* `some_complex_function_loss/c`

Each individual loss will still be subject to the `sample_weight` and `reduction` behavior as specified to `SomeComplexFunction`.

### Multiple Outputs + Labels
The `Model`'s constructor `loss` argument can accept a single `Loss`, a `list` or `dict` of losses, and even nested structures of the previous. However, in Elegy the form of `loss` is not strictly related to structure of input labels and outputs of the model. This is very different to Keras where each loss has to be matched with exactly one (label, output) pair. Elegy's method of dealing with multiple outputs and labels is super simple:

!!! Quote
    - `y_true` will contain the **entire** structure passed to `y`.
    - `y_pred` will contain the **entire** structure output by the `Module`.

This means there are no restrictions on how you structure the loss function. According to this rule Keras and Elegy behave the same when there is only one output and one label because there is no structure. Both frameworks will allow you to define something like: 

```python
model = Model(
    ...
    loss=elegy.losses.CategoricalCrossentropy(from_logits=True)
)
```
However, if you have many outputs and many labels, Elegy will just pass their structures to your loss and you will be able to do whatever you want by e.g. indexing these structures:

```python
class MyLoss(Elegy.Loss):
    def call(self, y_true, y_pred):
        return some_function(
            y_true["label_a"], y_pred["output_a"], y_true["label_b"]
        )

model = Model(
    ...
    loss=elegy.losses.MyLoss()
)
```
This example assumes the `y_true` and `y_pred` are dictionaries but they can also be tuples or nested structures. This strategy gives you maximal flexibility but come with the additional cost of having to implement your own loss function. 

### Keras-like behavior
While having this flexibility is good, there is a common scenario that Keras covers really well: what if you really just need one loss per (label, output) pair? In other words, how can we achieve equivalent behaviour of the following Keras code?

```python
class MyModel(keras.Model):
    def call(self, x):
        ...
        return {
            "key_a": key_a,
            "key_b": key_b,
            ...
        }
...
model.compile(
    loss={
        "key_a": keras.losses.BinaryCrossentropy(),
        "key_b": keras.losses.MeanSquaredError(),
        ...
    },
    loss_weights={
        "key_a": 10.0,
        "key_b": 1.0,
        ...
    },
)
```
To do this Elegy lets each `Loss` optionally filter / index the `y_true` and `y_pred` arguments based on a string key (for `dict`s) or integer key (for `tuple`s) in the constructor's `on` parameter:

```python
class MyModule:
    def call(self, x):
        ...
        return {
            "key_a": key_a,
            "key_b": key_b,
            ...
        }
...
model = elegy.Model(
    module=MyModule(),
    loss=[
        elegy.losses.BinaryCrossentropy(on="key_a", weight=10.0),
        elegy.losses.MeanSquaredError(on="key_b", weight=1.0),
        ...
    ]
)
```
This is almost exactly how Keras behaves except each loss is explicitly aware of which part of the output / label its supposed to attend to. The previous is roughly equivalent to manually indexing `y_true` and `y_pred` and passing the resulting value to the loss in question like this:

```python
model = elegy.Model(
    module=MyModule(),
    loss=[
        lambda y_true, y_pred: elegy.losses.BinaryCrossentropy(weight=10.0)(
            y_true=y_true["key_a"],
            y_pred=y_pred["key_a"],
        ),
        lambda y_true, y_pred: elegy.losses.MeanSquaredError(weight=1.0)(
            y_true=y_true["key_b"],
            y_pred=y_pred["key_b"],
        ),
        ...
    ]
)
```
!!! Note
    For the same reasons Elegy doesn't support the `loss_weights` parameter as defined in `keras.compile`. Nonetheless, each loss accepts a `weight` argument directly, as seen in the examples above, which you can use to recover this behavior.

## Metrics
Metrics behave exactly like losses except for one thing:

!!! Quote
    Metrics can hold state. 

As in Keras, Elegy metrics are cumulative so they update their internal state on every step. From a user's perspective this means that you should use the `self.add_parameter` and `self.update_parameter` hooks when implementing your own metrics.

Here is an example of a simple cumulative implementation of `Accuracy` which uses state hooks:

```python
class Accuracy(elegy.Metric):
    def call(self, y_true, y_pred):

        total = self.add_parameter("total", initializer=0, trainable=False)
        count = self.add_parameter("count", initializer=0, trainable=False)

        total += jnp.sum(y_true == y_pred)
        count += jnp.prod(y_true.shape)

        self.update_parameter("total", total)
        self.update_parameter("count", count)

        return total / count
```

For a more in-depth description of how Elegy's hook system works check out the [Module System](../module-system) guide.
