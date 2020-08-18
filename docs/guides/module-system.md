# The Module System

This is a guide to Elegy's underlying Module System. It will help get a better understanding of how
Elegy interacts with Jax at the lower level, certain details about the hooks system and how it
differs from other Deep Learning frameworks.

### Traditional Object Oriented Style
We will begin by exploring other frameworks define Modules / Layers. It is very common to use
Object Oriented architectures as backbones of Module systems as it helps frameworks keep
track of the parameters and states each module might require. Here we will create a some
very basic `Linear` and `MLP` modules which will seem very familiar:

```python
class Linear(elegy.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.w = elegy.get_parameter(
            "w",
            [x.shape[-1], self.n_out],
            initializer=elegy.initializers.RandomUniform(),
        )
        self.b = elegy.get_parameter("b", [n_out], initializer=elegy.initializers.RandomUniform())

    def call(self, x):
        return jnp.dot(x, self.w) + self.b

class MLP(elegy.Module):
    def __init__(self, n_in):
        self.linear1 = Linear(n_in, 64)
        self.linear2 = Linear(64, 32)
        self.linear3 = Linear(32, 1)

    def call(self, x):
        x = self.linear1(x)
        x = jax.nn.relu(x)
        x = self.linear2(x)
        x = jax.nn.relu(x)
        x = self.linear3(x)
        return x
```

Here we just defined a simple linear layer and used it inside a `MLP` with 3 layers. Pytorch and Keras 
users should feel very familiar with this type of code: we define parameters 
or other submodules in the `__init__` method, and use them during the `call` (forward) method.
Keras users users might complain that if we do things this way we loose the ability to do 
shape inference, but don't worry, we will fix that latter.

Fow now it is important to notice that here we use our first hook: `get_parameter`.

### Elegy Hooks
Hooks are a way in which we can manage state while preserving functional purity (in the end).
Elegy's hook system is ported and expanded from Haiku, but hook-based functional architectures in
other areas like web development have proven valuable, React Hooks being a recent notable success.

In Elegy we have the following list of hooks:


| Hook            | Description                                                                                                     |
| --------------- | --------------------------------------------------------------------------------------------------------------- |
| `get_parameter` | Gives us access to a trainable parameter.                                                                       |
| `get_state`     | Gives us access to some state. This is used in layers like `BatchNormalization` and in most of the metrics.     |
| `set_state`     | Lets us update a state. When used in conjunction with `get_state` it lets use express an iterative computation. |
| `next_rng_key`  | Gives us access to a unique `PRNGKey` we can pass to functions like `jax.random.uniform` and friends.           |
| `is_training`   | Tells use whether training is currently happening or not.                                                       |
| `add_loss`      | Lets us declare a loss in some intermediate layer.                                                              |
| `add_metric`    | Lets us declare a metric in some intermediate layer.                                                            |
| `add_summary`   | Lets us declare a summary in some intermediate layer.                                                           |

!!! Note
    If you use existing `Module`s you might not need to worry much about this, but keep these in might
    if you are developing your own custom modules.

### Module Hooks: Functional Style
In the initial example we used hooks in a very shy manner to replicate the behavior of
of other frameworks, now we will go beyond. The first thing we need to know is that:

!!! Quote
    **Modules are hooks**

This means that module _instantiation_ taps into the hook system, and that hooks are
aware of the module in which they are executing. In practice this will mean that we
will be able to move a lot of the code usually defined on the `__init__` method to
the `call` method:

```python
class Linear(elegy.Module):
    def __init__(self, n_out):
        super().__init__()
        self.n_out = n_out

    def call(self, x):
        w = elegy.get_parameter(
            "w",
            [x.shape[-1], self.n_out],
            initializer=elegy.initializers.RandomUniform(),
        )
        b = elegy.get_parameter("b", [self.n_out], initializer=jnp.zeros)

        return jnp.dot(x, w) + b

class MLP(elegy.Module):
    def call(self, x):
        x = Linear(64)(x)
        x = jax.nn.relu(x)
        x = Linear(32)(x)
        x = jax.nn.relu(x)
        x = Linear(1)(x)
        return x
```

What happened here? Lets decompose it into two parts. First we moved the `get_parameter` definitions
on the `Linear` module to the `call` method:

```python hl_lines="7 8 9 10 11"
class Linear(elegy.Module):
    def __init__(self, n_out):
        super().__init__()
        self.n_out = n_out

    def call(self, x):
        w = elegy.get_parameter(
            "w",
            [x.shape[-1], self.n_out],
            initializer=elegy.initializers.RandomUniform(),
        )
        b = elegy.get_parameter("b", [self.n_out], initializer=jnp.zeros)

        return jnp.dot(x, w) + b
```

As you see this not allows us to do shape inference since we have access
to the inputs when defining our parameters. Second, we also moved the instantiation
of the `Linear` modules in `MLP` from `__init__` to `call`:

```python hl_lines="3 5 7"
class MLP(elegy.Module):
    def call(self, x):
        x = Linear(64)(x)
        x = jax.nn.relu(x)
        x = Linear(32)(x)
        x = jax.nn.relu(x)
        x = Linear(1)(x)
        return x
```

Here we are using Modules as hooks. While it may _appear_ as if we are instantiating
3 new `Linear` modules on every `call`, Elegy is actually caching them behind the scenes
with the help of Python's metaclasses. There is one important rule you have to follow:

!!! Quote
    You must use Module hooks **unconditionally**

This moto comes from React and it means that the module always has to call the same amount 
of submodule hooks in the same order. For example the following code is invalid:

```python
def call(self, x):
    if x.shape[0] > 5:
        x = elegy.nn.Conv2D(32, [3, 3])(x)
        x = elegy.nn.Linear(48, [3, 3])(x)
    else:
        x = elegy.nn.Linear(48, [3, 3])(x)
        x = elegy.nn.Conv2D(32, [3, 3])(x)
    return x
```
Here `Linear` and `Conv2D` are dangerously swapped based on some condition. If you want to
do this you can just clare them unconditionally and use them inside the condition:

```python
def call(self, x):
    linear = elegy.nn.Linear(48, [3, 3])
    conv2d = elegy.nn.Conv2D(32, [3, 3])

    if x.shape[0] > 5:
        x = conv2d(x)
        x = linear(x)
    else:
        x = linear(x)
        x = conv2d(x)
    return x
```

### init & apply
This functional freedom inside Modules comes at a cost outside of them
which is that you cannot call the top-level module directly. If you try to run this code:

```python
x = np.random.uniform(size=(15, 3))
mlp = MLP()
y = mlp(x)
```

You will get the following error:

> ValueError: Cannot execute `call` outside of a `elegy.context`

In practice this means that you will have to use the methods `init` and `apply`
to manage your modules:

```python
x = np.random.uniform(size=(15, 3))
mlp = MLP()
rngs = elegy.PRNGSequence(42)
mlp.init(rng=next(rngs))(x)
y_pred, ctx = mlp.apply(rng=next(rngs))(x)
```

A lot is happening here so lets unpack it. First we used `Module.init(...)` and passed
it an `rng` key. 

```python hl_lines="4"
x = np.random.uniform(size=(15, 3))
mlp = MLP()
rngs = elegy.PRNGSequence(42)
mlp.init(rng=next(rngs))(x)
y_pred, ctx = mlp.apply(rng=next(rngs))(x)
```

This key is necessary because `Linear` uses `elegy.initializers.RandomUniform`
which requires a random key to initialize our weights. `init` takes in some parameters and
returns callable which expect the same arguments as `call` and will initialize our module. 
Next we use `Module.apply(...)` very similarly but this time we get back some predictions and
a context:

```python hl_lines="5"
x = np.random.uniform(size=(15, 3))
mlp = MLP()
rngs = elegy.PRNGSequence(42)
mlp.init(rng=next(rngs))(x)
y_pred, ctx = mlp.apply(rng=next(rngs))(x)
```

Right now we won't use this context object but its useful to know that this object will
collect most of the information given by hooks like `get_parameter`, `get_state`, `add_loss`,
`add_metric`, etc.

### Hooks Preserve References
In our `MLP` class we where able to create the `Linear` modules at their call site, this
simplified our code a lot but we've seem to lost the reference to these modules. Having 
reference to other modules is critical for being able to e.g. easily compose modules that might be
trained separately like in transfer learning, or being able to easily decompose / extract a sub-module
and use it separately like when using the decoder of a VAE by itself to generate new samples.

Because of this, Elegy actually assigns all submodules, parameters, and states as properties
of the module:

```python hl_lines="5 7"
x = np.random.uniform(size=(15, 3))
mlp = MLP()
rngs = elegy.PRNGSequence(42)
mlp.init(rng=next(rngs))(x)
linear, linear_1, linear_2 = mlp.linear, mlp.linear_1, mlp.linear_2
y_pred, ctx = mlp.apply(rng=next(rngs))(x)
assert linear is mlp.linear and linear_1 is mlp.linear_1 and linear_2 is mlp.linear_2
```
As you see we were able to access all the linear layer references. More over, we verified that
these reference don't change during execution. Each submodule gets assigned to a 
a unique field name based on its class name and order of creation. You can
customize this name by using the `name` argument available in the `Module`'s constructor.

### Managing Parameters and States

```python hl_lines="4 5"
x = np.random.uniform(size=(15, 3))
mlp = MLP()
rngs = elegy.PRNGSequence(42)
parameters, states = mlp.init(rng=next(rngs))(x)
y_pred, ctx = mlp.apply(parameters=parameters, states=states, rng=next(rngs))(x)
```

### Low-level Training Loop
```python
def loss(parameters, rng, x, y):
    y_pred, ctx = mlp.apply(parameters=parameters, states=states, rng=rng, training=True)(x)
    return jnp.mean(jnp.square(y - y_pred))


@jax.jit
def update(parameters, rng, x, y):
    gradients = jax.grad(loss)(parameters, rng, x, y)
    new_parameters = jax.tree_multimap(
        lambda p, g: p - 0.001 * g, parameters, gradients
    )
    return new_parameters


x = np.random.uniform(size=(15, 3))
y = np.random.uniform(size=(15, 1))
mlp = MLP()
parameters, states = mlp.init(rng=next(rngs))(x)

for step in range(1000):
    parameters = update(parameters, next(rngs), x, y)

mlp.set_parameters(parameters)
```

```python hl_lines="1 8"
def loss(parameters, rng, x, y):
    y_pred, ctx = mlp.apply(parameters=parameters, states=states, rng=rng, training=True)(x)
    return jnp.mean(jnp.square(y - y_pred))


@jax.jit
def update(parameters, rng, x, y):
    gradients = jax.grad(loss)(parameters, rng, x, y)
    new_parameters = jax.tree_multimap(
        lambda p, g: p - 0.001 * g, parameters, gradients
    )
    return new_parameters
```

```python hl_lines="7 12"
def loss(parameters, rng, x, y):
    y_pred, ctx = mlp.apply(parameters=parameters, states=states, rng=rng, training=True)(x)
    return jnp.mean(jnp.square(y - y_pred))


@jax.jit
def update(parameters, rng, x, y):
    gradients = jax.grad(loss)(parameters, rng, x, y)
    new_parameters = jax.tree_multimap(
        lambda p, g: p - 0.001 * g, parameters, gradients
    )
    return new_parameters
```

```python hl_lines="6 8"
x = np.random.uniform(size=(15, 3))
y = np.random.uniform(size=(15, 1))
mlp = MLP()
parameters, states = mlp.init(rng=next(rngs))(x)

for step in range(1000):
    parameters = update(parameters, next(rngs), x, y)

mlp.set_parameters(parameters)
```
### High Level Equivalent
```python
model = elegy.Model(
    module=elegy.nn.Sequential(
        lambda: [
            elegy.nn.Linear(64),
            jax.nn.relu,
            elegy.nn.Linear(32),
            jax.nn.relu,
            elegy.nn.Linear(1),
        ]
    ),
    loss=elegy.losses.MeanSquaredError(),
)

model.fit(
    x=np.random.uniform(size=(15, 3)),
    y=np.random.uniform(size=(15, 1)),
    batch_size=15,
    epochs=1000,
)
```

