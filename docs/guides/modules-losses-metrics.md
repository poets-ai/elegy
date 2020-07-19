# Modules, Losses, and Metrics

This guide goes into depth on how modules, losses and metrics work in Elegy and how to create your own. One of our goals with Elegy was to solve Keras restrictions around the type of losses and metrics you can define.

When creating a complex model with multiple outputs with Keras, say `output_a` and `output_b`, you are forced to define losses and metrics per-output only:

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
This very restrictive, in particular it doesn't allow the following:

1. Losses and metrics that combine multiple outputs with multiple labels.
2. A single loss/metrics based on multiple outputs (a especial case of the previous).
3. Losses and metrics that depend on the inputs of the model.

Most of these are usually solvable by [concatenating the outputs / labels](https://stackoverflow.com/a/57030727/2118130) or passing the inputs as labels. However it is clear that these solution are hacky at best and depending on the problem they can be insufficient. 

## Dependency Injection
Elegy solves the previous problems by instruducting a _dependency injection_ mechanism that allows the user to express complex function by simply declaring the variables it wants to use **by their name**. The following parameters are available for the different callables you pass to Elegy:


 | parameter       | Description                                                    | Module | Metric | Loss |
 | --------------- | -------------------------------------------------------------- | ------ | ------ | ---- |
 | `x`             | Inputs of the model corresponding to the `x` argument of `fit` | *      | x      | x    |
 | `y_true`        | The input labels corresponding to the `y` argument of `fit`    |        | x      | x    |
 | `y_pred`        | Outputs of the model                                           |        | x      | x    |
 | `sample_weight` | Importance of each sample                                      |        | x      | x    |
 | `class_weight`  | Importance of each class                                       |        | x      | x    |
 | `is_training`   | Whether training is currently in progress                      | x      | x      | x    |
 | `params`        | The learnable parameters of the model                          |        | x      | x    |
 | `state`         | The non-learnable parameters of the model                      |        | x      | x    |


!!! Note
    The content of `x` is technically passed to the model's `Module` but the parameter name _x_ will bare no special meaning there.


## Modules
Modules define the architecture of the network, their primary task (Elegy terms) is transforming `x` into `y_pred`. To make it easy to consume the content of `x` Elegy has some special but very simple rules on how the signature of any `Module` can be structured:

If `x` is a `tuple` then `x` will be expanded positional arguments a.k.a. `*args`, this means that the module will have define EXACTLY as many arguments as there are inputs. For example:
  
```python hl_lines="2 10"
class SomeModule(elegy.Module):
    def call(self, m, n):
        ...

...

a, b = get_inputs()

model.fit(
    x=(a, b),
    ...
)
```
In this case `a` is passed as `m` and `b` is passed as `n`.

On the other hand, if `x` is a `dict` then `x` will be expanded as keyword arguments a.k.a. `**kwargs`, in this case the module can optionally request any key defined in `x` as an argument. For example:

```python hl_lines="2 10"
class SomeModule(elegy.Module):
    def call(self, n):
        ...

...

a, b = get_inputs()

model.fit(
    x={"m": a, "n": b},
    ...
)
```
Here `n` is request by name and you get as input its value `b`, and `m` is safely ignored.

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
    ...
)
```
Here the input `y` is passed as `y_true` as stated previously. However, if for example you want to build an autoencoder you don't actually need `y` since you just want to reconstruct `x`, therefore it makes perfect sense to just request `x` and Elegy lets you do exactly that:

```python hl_lines="2"
class AutoEncoderLoss(elegy.Loss): 
    def call(self, x, y_pred):
        return jnp.mean(jnp.square(x - y_pred), axis=-1)

...

X_train, _ = get_inputs()

model.fit(
    x=X_train
    ...
)
```
Here we only used `x` instead of `y_true` to define the loss as the math usually tells use, therefore no `y` was required on `fit`.

!!! Note
    In this case you could have easily just passed `y=X_train` and reused the previous `MSE` definition, avoiding the creation of redundant labels is good in general and being explicit about e.g. the function using `x` might even self-document its behaviour.

### Partitioning a loss
If you have a complex loss function that is just a sum of different subparts but that you probably have to compute together to e.g. reuse some computation, you might define something like this:
```python
class SomeComplexFunction(elegy.Loss): 
    def call(self, x, y_true, y_pred, params, ...):
        ...

        return a + b + c
```
Purely for logging purposes you can instead return a `dict` of these losses:

```python
class SomeComplexFunction(elegy.Loss): 
    def call(self, x, y_true, y_pred, params, ...):
        ...
        return {
            "a": a,
            "b": b,
            "c": c,
        }
```
Elegy will use this information to show you each loss separate in the logs / Tensorboard with the names:
* `some_complex_function_loss/a`
* `some_complex_function_loss/b`
* `some_complex_function_loss/c`

### Multiple Outputs + Labels
The models constructor `loss` argument can accept a single `Loss`, a `list` or `dict` of losses, and even nested structures of the previous, yet the form of `loss` is not strictly related to structure or numbers of labels and outputs of the model. This is very different to Keras where each loss has to be match with exactly 1 label and 1 output. Elegy's method of dealing with multiple outputs and labels is super simple:

!!! Quote
    `y_true` and `y_pred` will **always** be passed to each loss exactly as they are defined

This means there are no restrictions on how you structure the loss function. According to this rule, for the simplest of cases where there is only 1 output and 1 label, Keras and Elegy will behave the same because there is no structure: 

```python
model = Model(
    ...
    loss=elegy.losses.CategoricalCrossentropy(from_logits=True)
)
```
But if you have many outputs and many labels Elegy will just pass them to you and you can just define your loss function by using indexing their stuctures:

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
This example assumes they are dictionaries but they can also be tuples. This gives you maximal flexibility but come at the additional cost of having to implement a custom loss function. 

### Keras-like behaviour
While these example show Elegy's flexibility, there is an inbetween scenario that Keras covers really well: what if you really just need 1 loss per (label, output) pair? For example the equivalent of:

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
    }
)
```
Elegy recovers this behaviour by letting each `Loss` filter (or rather index) `y_true` and `y_pred` based on a string key in the case of dictionaries or int key in the case of tuples using the constructors `on` parameter:

```python
class MyModule(elegy.Module):
    def call(self, x):
        ...
        return {
            "key_a": key_a,
            "key_b": key_b,
            ...
        }
...
model = elegy.Model(
    module=MyModule.defer(),
    loss=[
        keras.losses.BinaryCrossentropy(on="key_a"),
        keras.losses.MeanSquaredError(on="key_b"),
        ...
    ]
)
```
This is almost exactly how Keras behaves except each loss is explicitly aware of which part of the output / label its suppose to attend to. The previous if roughly equivalent to manually indexing `y_true` and `y_pred` and passing the resulting value to the loss with like this:

```python
model = elegy.Model(
    module=MyModule.defer(),
    loss=[
        lambda y_true, y_pred: keras.losses.BinaryCrossentropy()(
            y_true=y_true["key_a"],
            y_pred=y_pred["key_a"],
        ),
        lambda y_true, y_pred: keras.losses.MeanSquaredError()(
            y_true=y_true["key_b"],
            y_pred=y_pred["key_b"],
        ),
        ...
    ]
)
```

## Metrics
Metrics behave very similar to losses, everything said about losses previously about losses holds for metrics except for one thing: metrics can hold state. As in Keras, Elegy metrics are cummulative metrics which update their internal state on every step. From a user perspective this a couple of things:

1. Metrics are implemented using Haiku `Module`s, this means that you can't instantiate them normally outside of Haiku, hence the `lambda` / `dered` trick.
2. You can use `hk.get_state` and `hk.set_state` when implementing you own metrics.

Here is an example of a simple implementation of Accuracy:

```python
class Accuracy(elegy.Metric):
    def call(self, y_true, y_pred):

        total = hk.get_state("total", [], jnp.zeros)
        count = hk.get_state("count", [], jnp.zeros)

        total += jnp.sum(y_true == y_pred)
        count += jnp.prod(y_true.shape)

        hk.set_state("total", total)
        hk.set_state("count", count)

        return total / count
```


## A little secret
We think users should use the base classes provided by Elegy (Module, Loss, Metric) for convenience, but the fact is that Elegy also accepts ordinary callables. Being true to Haiku and Jax in general, you can just use functions, however you can run into trouble with Haiku due to not scoping you computation inside Modules.