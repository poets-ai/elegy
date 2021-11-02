# pred_step
The `pred_step` method computes the predictions of the main model, by overriding this method you can directly influence what happens during `predict`.

### Inputs
Any of following input arguments are available for `pred_step`:

| name           | type     |                                          |
| :------------- | :------- | :--------------------------------------- |
| `x`            | `Any`    | Input data                               |
| `states`       | `States` | Current state of the model               |
| `initializing` | `bool`   | Whether the model is initializing or not |
| `training`     | `bool`   | Whether the model is training or not     |

You must request the arguments you want by **name**.

### Outputs
`pred_step` must output a tuple with the following values:

| name     | type     |                              |
| :------- | :------- | :--------------------------- |
| `y_pred` | `Any`    | The predictions of the model |
| `states` | `States` | The new state of the model   |


### Callers
| method         | when                   |
| :------------- | :--------------------- |
| `predict`      | always                 |
| `test_step`    | default implementation |
| `summary_step` | default implementation |

### Examples
If for some reason you wish to create a pure jax / Module-less model, you can define your own Model that implements `pred_step` like this:

```python
class LinearClassifier(elegy.Model):
    def pred_step(self, x, y_true, states, initializing):  
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
        y_pred = jnp.dot(x, w) + b

        return y_pred, states.update(net_params=(w, b))

model = LinearClassifier(
    optimizer=optax.adam(1e-3),
    loss=elegy.losses.Crossentropy(),
    metrics=elegy.metrics.SparseCategoricalAccuracy(),
)

model.fit(
    x=X_train,
    y=y_train,
    epochs=100,
    batch_size=64,
)
```
Here we implement the same `LinearClassifier` from the [basics](./basics) section but we extracted the definition of the model to `pred_step` and we let the basic implementation of `test_step` take care of the `loss` and `metrics` which we provide to the `LinearClassifier`'s constructor.

### Default Implementation
The default implementation of `pred_step` does the following:

* Calls `api_module.init` or `api_module.apply` depending on state of `initializing`. `api_module` of type `GeneralizedModule` is a wrapper over the `module` object passed by the user to the `Model`s constructor.