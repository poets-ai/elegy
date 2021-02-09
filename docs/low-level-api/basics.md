# Low-level API
Elegy's low-level API allows you to override some core methods in `Model` that specify what happens during training, inference, etc. This approach is perfect when you want to do things that are hard or simply not possible with the high-level API as it gives you the flexibility to do anything inside these methods as long as you return the expected types. 


### Methods
This is the list of all the overrideable methods:

| Caller Methods                       | Overridable Method |
| :----------------------------------- | :----------------- |
| - `predict` <br>- `predict_on_batch` | `pred_step`        |
| - `evaluate`<br>- `test_on_batch`    | `test_step`        |
|                                      | `grad_step`        |
| - `fit`<br>- `train_on_batch`        | `train_step`       |
| - `summary`                          | `summary_step`     |

Each overrideable method has a default implementation which is what gives rise to the high-level API, the default implementation almost always implements a method in term of another in this manner:

```
pred_step ⬅ test_step ⬅ grad_step ⬅ train_step
pred_step ⬅ summary_step
```
This allows you to e.g. override `test_step` and still be able to use use `fit` since `train_step` (called by `fit`) will call your `test_step` via `grad_step`. It also means that e.g. if you implement `test_step` but not `pred_step` there is a high chance both `predict` and `summary` will not work as expected since both depend on `pred_step`. 

### Example
Each overrideable methods takes some input + state, performs some `jax` operations + updates the state, and returns some outputs + the new state. Lets see a simple example of a linear classifier using `test_step`:

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

As you see here we perform everything from parameter initialization, modeling, calculating the main loss, and logging some metrics. Notes:

* The `states` argument of type `elegy.States` is an immutable Mapping which you add / update fields via its `update` method.
* `net_params` is one of the names used by the default implementation, check the [States](./states.md) guid for more information.
* `initializing` tells you whether you to initialize your parameters or fetch them from `states`, if you are using a Module framework this usually tells you whether to call `init` or `apply`.
* `test_step` returns 3 specific outputs, you should check the docs for each method to know what to return.
