# Low-level API
Elegy's low-level API allows you to override some core methods in `Model` that specify what happens during training, inference, etc. This approach is perfect when you want to do things that are hard or simply not possible with the high-level API as it gives you the flexibility to do anything inside these methods as long as you return the expected types. 


### Methods
This is the list of all the overrideable methods:

| Caller     | Method         |
| :--------- | :------------- |
| `predict`  | `pred_step`    |
| `evaluate` | `test_step`    |
|            | `grad_step`    |
| `fit`      | `train_step`   |
| `init`     | `init_step`    |
| `summary`  | `summary_step` |
|            | `states_step`  |
|            | `jit_step`     |

Each method has a default implementation which is what gives rise to the high-level API.

### Example
Most overrideable methods take some input & state, perform some `jax` operations & updates the state, and returns some outputs & the new state. Lets see a simple example of a linear classifier using `test_step`:

```python
class LinearClassifier(elegy.Model):
    def test_step(self, x, y_true, states, initializing):  
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

As you see here we perform everything from parameter initialization, modeling, calculating the main loss, and logging some metrics. Some notes about the previous example:

* The `states` argument of type `elegy.States` is an immutable Mapping which you add / update fields via its `update` method.
* `net_params` is one of the names used by the default implementation, check the [States](./states.md) guide for more information.
* `initializing` tells you whether to initialize the parameters of the model or fetch the current ones from `states`, if you are using a Module framework this usually tells you whether to call `init` or `apply`.
* `test_step` should returns 3 specific outputs (`loss`, `logs`, `states`), you should check the docs for each method to know what to return.
