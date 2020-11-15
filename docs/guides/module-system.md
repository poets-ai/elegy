# The Module System

This is a guide to Elegy's underlying Module System. It will help get a better understanding of how
Elegy interacts with Jax at the lower level, certain details about the hooks system and how it
differs from other Deep Learning frameworks.

### Traditional Object Oriented Style
We will begin by exploring how other frameworks define Modules / Layers. It is very common to use
Object Oriented architectures as backbones of Module systems as it helps frameworks keep
track of the parameters and states each module might require. Here we will create some
very basic `Linear` and `MLP` modules which will seem very familiar:

```python
class Linear(elegy.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.w = self.add_parameter(
            "w",
            [x.shape[-1], self.n_out],
            initializer=elegy.initializers.RandomUniform(),
        )
        self.b = self.add_parameter("b", [n_out], initializer=elegy.initializers.RandomUniform())

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
Keras users might complain that if we do things this way we loose the ability to do 
shape inference, but don't worry, we will fix that latter.

Fow now it is important to notice that here we use our first hook: `add_parameter`.

### Elegy Hooks
Hooks are a way in which we can manage state while preserving functional purity (in the end).
Elegy's hook system is ported and expanded from Haiku, but hook-based functional architectures in
other areas like web development have proven valuable, React Hooks being a recent notable success.

In Elegy we have the following list of hooks:


| Hook                 | Description                                                                                           |
| -------------------- | ----------------------------------------------------------------------------------------------------- |
| `self.add_parameter` | Gives us access to trainable and non-trainable parameters.                                            |
| `elegy.add_loss`     | Lets us declare a loss from some intermediate module.                                                 |
| `elegy.add_metric`   | Lets us declare a metric in some intermediate module.                                                 |
| `elegy.add_summary`  | Lets us declare a summary in some intermediate module.                                                |
| `elegy.training`     | Tells us whether training is currently happening or not.                                              |
| `elegy.next_rng_key` | Gives us access to a unique `PRNGKey` we can pass to functions like `jax.random.uniform` and friends. |

!!! Note
    If you use existing `Module`s you might not need to worry much about these hooks, but keep them in mind
    if you are developing your own custom modules.

### Module Hooks: Functional Style
In the initial example we used hooks in a very shy manner to replicate the behavior of
of other frameworks, now we will go beyond. The first thing we need to know is that:

!!! Quote
    **Modules are hooks**

This means that module _instantiation_ taps into the hook system, and that hooks are
aware of the module in which they are executing in. In practice this will mean that we
will be able to move a lot of the code defined on the `__init__` method to
the `call` method:

```python
class Linear(elegy.Module):
    def __init__(self, n_out):
        super().__init__()
        self.n_out = n_out

    def call(self, x):
        w = self.add_parameter(
            "w",
            [x.shape[-1], self.n_out],
            initializer=elegy.initializers.RandomUniform(),
        )
        b = self.add_parameter("b", [self.n_out], initializer=jnp.zeros)

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

What happened here? Lets decompose it into two parts. First we moved the `add_parameter` definitions
on the `Linear` module to the `call` method:

```python hl_lines="7 8 9 10 11"
class Linear(elegy.Module):
    def __init__(self, n_out):
        super().__init__()
        self.n_out = n_out

    def call(self, x):
        w = self.add_parameter(
            "w",
            [x.shape[-1], self.n_out],
            initializer=elegy.initializers.RandomUniform(),
        )
        b = self.add_parameter("b", [self.n_out], initializer=jnp.zeros)

        return jnp.dot(x, w) + b
```

As you see this allows us to do shape inference since we have access
to the inputs when defining our parameter's shape. Second, we also moved the instantiation
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
with the help of Python metaclasses. There is one important rule you have to follow:

!!! Quote
    You must use hooks **unconditionally**

This motto comes from React and it means that the module always has to call the same amount
of hooks, and for module hooks specifically they must be called in the same order. For example the following code is invalid:

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

### Hooks Preserve References
In our `MLP` class we where able to create the `Linear` modules at their call site, this
simplified our code a lot but we've seem to lost the reference to these modules. Having 
reference to other modules is critical for being able to e.g. easily compose modules that might be
trained separately like in transfer learning, or being able to easily decompose / extract a sub-module
and use it separately like when using the decoder of a VAE by itself to generate new samples.

Because of this, Elegy actually assigns all submodules, parameters, and states as properties
of the module:

```python hl_lines="5 8"
x = np.random.uniform(size=(15, 3))
mlp = MLP()

mlp(x)
linear, linear_1, linear_2 = mlp.linear, mlp.linear_1, mlp.linear_2

y_pred = mlp(x)
assert linear is mlp.linear and linear_1 is mlp.linear_1 and linear_2 is mlp.linear_2
```
As you see we were able to access all the linear layer references. More over, we verified that
these reference don't change during execution. Each submodule gets assigned to a 
a unique field name based on its class name and order of creation. You can
customize this name by using the `name` argument available in the `Module`'s constructor.

### Low-level Training Loop

A big theme in Jax is that state and computation are separate, this is a requirement
because in order for combinators like `jax.grad` and `jax.jit` to work you need pure functions. 
Elegy as you've seen is object oriented so additional effort ir required to properly convert all 
the global states and `Module` parameters an inputs to a function so Jax can track them. To achieve 
Elegy implements its own `jit` and `value_and_grad` function wrappers that handle this for you.

Lets create a low level training loop using the previous definition `MLP` along with these functions:

```python
x = np.random.uniform(size=(15, 3))
y = np.random.uniform(size=(15, 1))
mlp = MLP()

def loss_fn(x, y):
    y_pred = mlp(x)
    return jnp.mean(jnp.square(y - y_pred))

def update(x, y):
    loss, gradients = elegy.value_and_grad(loss_fn, modules=mlp)(x, y)
    parameters = mlp.get_parameters(trainable=True)
    new_parameters = jax.tree_multimap(
        lambda p, g: p - 0.01 * g, parameters, gradients
    )

    mlp.set_parameters(new_parameters)

    return loss

update_jit = elegy.jit(update, modules=mlp)

for step in range(1000):
    loss = update_jit(x, y)
    print(step, loss)
```

Here we created the functions `loss_fn` and `update`, plus a minimal training loop.
Loss `loss_fn` calculate the Mean Square Error while `update` uses `value_and_grad` to calculate the gradient
of the loss with respect to the trainable parameters of `mlp`.

```python hl_lines="2"
def update(x, y):
    loss, gradients = elegy.value_and_grad(loss_fn, modules=mlp)(x, y)
    parameters = mlp.get_parameters(trainable=True)
    new_parameters = jax.tree_multimap(
        lambda p, g: p - 0.01 * g, parameters, gradients
    )

    mlp.set_parameters(new_parameters)

    return loss
```

After that we just use `tree_multimap` to implement Gradient Descent 
and get our `new_parameters` and then use the `set_parameters` method our 
`Module` to update its state.

```python hl_lines="4 5 6 8"
def update(x, y):
    loss, gradients = elegy.value_and_grad(loss_fn, modules=mlp)(x, y)
    parameters = mlp.get_parameters(trainable=True)
    new_parameters = jax.tree_multimap(
        lambda p, g: p - 0.01 * g, parameters, gradients
    )

    mlp.set_parameters(new_parameters)

    return loss
```

Having our update function we can use `elegy.jit` to create
an optimized version of our computation and create a minimal
training loop.

```python hl_lines="1"
update_jit = elegy.jit(update, modules=mlp)

for step in range(1000):
    loss = update_jit(x, y)
    print(step, loss)
```

Notice that even though we are jitting `update` which has the `set_parameters` side effect
(normally forbidden in Jax), learning is happening because `update_jit` automatically keeps track
of changes to the parameters of `mlp` and updates them for us. Something similar is done 
in `elegy.value_and_grad` as you saw previously.

!!! Note
    Elegy has 2 types states: module state for the parameters of
    models and global state where Elegy keeps track of certain variables
    like an RNG for convenience. Elegy's `jit` behaves just like its
    Jax counterpart except that its aware of the changes in state such that:

    * Jax properly recompiles if something changes.
    * The jitted function behaves *similar* to its eager version in that it propagates changes in state inwards and outwards (this only applies to Elegy states, not arbitrary side effects).


### High Level Equivalent

If all this seems a bit too manual for you don't worry, you can can easily express 
all the previous in a few lines of code using an `elegy.Model`:

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

