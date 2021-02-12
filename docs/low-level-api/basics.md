# Low-level API
Elegy's low-level API allows you to override some core methods in `Model` that specify what happens during training, inference, etc. This approach is perfect when you want to do things that are hard or simply not possible with the high-level API as it gives you the flexibility to do anything inside these methods with few restrictions. Most overrideable method do the following: 

1. Takes some inputs and state
2. Performs some `jax` operations and updates the state
3. Returns some outputs and the new state. 
 
Lets see a simple example of this API by creating a linear classifier by overrding `test_step`:

```python
class LinearClassifier(elegy.Model):
    def test_step(self, x,  y_true,  states, initializing) -> elegy.TestStep:  
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
        logs = dict(accuracy=accuracy, loss=loss)

        # update states
        states = states.update(net_params=(w, b))

        return loss, logs, states
```
There is a lot happening here we will explain later but as you see here we perform everything from parameter initialization, modeling, calculating the main loss, and logging some metrics. To actually use it you just create an instance and use the Model API as you normally would:

```python
model = LinearClassifier(
    optimizer=optax.adam(1e-3)
)

model.fit(
    x=X_train,
    y=y_train,
    epochs=100,
    batch_size=64,
)
```


### Methods
Most high-level API methods have an associated low-level API method it calls internally that you can override, here is the list of methods:

| High Level | Low Level      |
| :--------- | :------------- |
| `predict`  | `pred_step`    |
| `evaluate` | `test_step`    |
|            | `grad_step`    |
| `fit`      | `train_step`   |
| `summary`  | `summary_step` |
| `init`     | `init_step`    |

So for example if you wanted to customize what happens during `fit` you can customize the `train_step` method. Given this you might be wondering 

!!! question
    Why did we override `test_step` in the example if we called `fit`?

The reason is that the default implementation has the following call structure:

```
 summary        predict     evaluate                   fit
    ⬇️              ⬇️           ⬇️                        ⬇️
summary_step ➡️ pred_step ⬅ test_step ⬅ grad_step ⬅ train_step
```
So what happened was this:
* `fit` called `train_step` to perform the optimization step
* `train_step` called `grad_step` to get the gradients needed for the optimizer
* `grad_step` called `test_step` to get the loss needed to compute the gradients

So in the end our code was called and we got the `fit` for free 🥳. However you could have also notice the following:

!!! warning
    We didn't implement `pred_step`.

This is not bad per-se but we also didn't provide a `module` to the `LinearClassifier`'s constructor (because we bypassed this) so if you call `predict` or `summary` you will infact get an error telling you this. This means that you should be aware of what methods you are actually supporting using the low-level API.

### Implementation Details
Lets review some of the details in the example so you can get a better sense of how on you own. We will just mention some of the Elegy-specific things and leave the modeling details out.

The first thing to note is that the `states` argument of type `elegy.States` is an immutable `Mapping` to which carries all the states you need including the parameters of the network, optimizer states, rng states, etc.

```python hl_lines="2"
class LinearClassifier(elegy.Model):
    def test_step(self, x,  y_true,  states, initializing) -> elegy.TestStep:  
        x = jnp.reshape(x, (x.shape[0], -1)) / 255

        # initialize or use existing parameters
        if initializing:
            w = jax.random.uniform(
                jax.random.PRNGKey(42), shape=[np.prod(x.shape[1:]), 10]
            )
            b = jax.random.uniform(jax.random.PRNGKey(69), shape=[1])
```

The `initializing` argument tells you whether you to should initialize your parameters or rather fetch them from `states`:

```python hl_lines="4"
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
```
If you are using a Module framework (Flax, Haiku, or `elegy.Module`) this usually tells you whether to should call `init` vs `apply`. The `net_params` is one of the names used by the default implementation, while its not necessary to use the same names its mandatory if you want to reuse certain default methods that expect specific names. 

```python hl_lines="7 21"
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
logs = dict(accuracy=accuracy, loss=loss)

# update states
states = states.update(net_params=(w, b))
```

Check the [States](./states.md) guide for more information. Finally notice that `test_step` returns a tuple with 3 specific outputs:

```python hl_lines="12"
# categorical crossentropy loss
labels = jax.nn.one_hot(y_true, 10)
loss = jnp.mean(-jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1))
accuracy=jnp.mean(jnp.argmax(logits, axis=-1) == y_true)

# metrics
logs = dict(accuracy=accuracy, loss=loss)

# update states
states = states.update(net_params=(w, b))

return loss, logs, states
```
All methods returns `states` as last element but each methods has a different set of outputs, you should check the docs for each method to know what is expected.
