# Low-level API
Elegy's low-level API gives you more power and flexibility by allowing you to specify the core computation that methods like `fit`, `predict`, and `evaluate` perform. To use the low-level API instead of just passing a couple of arguments to `Model` you actually create your own subclass of it and override some methods. 


### Methods
This is the list of all the overrideable methods:

| Method       | Task                                                       | Callers                                  | Default Calls |
| :----------- | :--------------------------------------------------------- | :--------------------------------------- | ------------- |
| `pred_step`  | Calculate the predictions of the model.                    | `predict`, `summary`, `predict_on_batch` |               |
| `test_step`  | Calculate the losses en metrics.                           | `evaluate`, `test_on_batch`              | `pred_step`   |
| `grad_step`  | Calculate the gradient for the network's parameters.       |                                          | `test_step`   |
| `train_step` | Update the parameters of the model based on the gradients. | `fit`, `train_on_batch`                  | `grad_step`   |

For each method you should be aware of its "Callers" -methods from `Model` class that directly calls it- and the "Default Calls" -a dependent method it calls on the default implementation. The default implementation has this simple dependency graph:
```
pred_step ⬅ test_step ⬅ grad_step ⬅ train_step
```
which makes it such that e.g. even if you just override `test_step` to define how the loss is calculated you can still use `fit` because the default implementation of `train_step` would call your custom implementation of `test_step` via `grad_step`.

### Example
Lets see a simple example of a linear classifier using `test_step`:

```python
class LinearClassifier(elegy.Model):
    def test_step(self, x,  y_true,  states, initializing):  
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

        return loss, logs, states.update(rng=rng, net_params=(w, b))
```

As you see here we perform everything from parameter initialization, modeling, calculating the main loss, and logging some metrics. To actually use this just have to define an instance and after that you can use it as always, e.g:

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

#### States
The low-level API provides a simple immutable / functional state management system via the `states: elegy.States` parameter passed all methods. `elegy.States` is a `NamedTuple` that contains all the states needed in `Model`, each method accepts the current state and returns the next. As you can see in the last line of the example, the `States.update` method gives you a simple way of updating the states tuple without having to manually copy all its other parameters.

### Composing Methods