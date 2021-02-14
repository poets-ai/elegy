# pred_step
This method is tasked with taking the input data and calculatin the predictions. Implementing this lets you 

#### Example

```python
class LinearClassifier(elegy.Model):
    def pred_step(self, x,  states, initializing) -> elegy.TestStep:  
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

        # update states
        states = states.update(net_params=(w, b))

        return logits, states

model = LinearClassifier()

model.init(x=X_train)

preds = model.predict(x=X_train, batch_size=32)
```

### Inputs
The following inputs are available for `pred_step`:

| name           | types    | description                              |
| :------------- | :------- | :--------------------------------------- |
| `x`            | `Any`    | Input data                               |
| `states`       | `States` | Current state of the model               |
| `initializing` | `bool`   | Whether the model is initializing or not |
| `training`     | `bool`   | Whether the model is training or not     |

Thanks to Dependency Injection you can request only the arguments you need by name. 

### Output
`pred_step` must output a `tuple` with the following values:

| name     | types    | description                  |
| :------- | :------- | :--------------------------- |
| `y_pred` | `Any`    | The predictions of the model |
| `states` | `States` | The new state of the model   |


### Callers

| method             |                    |
| :----------------- | :----------------- |
| `predict`          |                    |
| `predict_on_batch` |                    |
| `test_step`        | unless overwritten |
| `summary_step`     | unless overwritten |

### Supporting the Default Implementation

If want to optionally support some of the behavior from the default implementation you can do the following:

* Use the `api_module` attribute to support the `module` passed by the user, `api_module` is a `GeneralizedModule` which contains an `init` and `apply` methods:

```python
class MyModel(elegy.Model):
    def pred_step(self, x,  states, initializing, training) -> elegy.TestStep:  
        x = jnp.reshape(x, (x.shape[0], -1)) / 255

        if initializing:
            model_fn = self.api_module.init(states.rng)
        else:
            model_fn = self.api_module.apply(
                params=states.net_params,
                states=states.net_states,
                training=training,
                rng=states.rng,
            )

        # model
        logits, net_params, net_states = model_fn(x)

        # update states
        states = states.update(net_params=(w, b))

        return logits, states

model = LinearClassifier(
    module=linen.Dense(10),
    optimizer=optax.adam(1e-3)
)

model.init(x=X_train, y=y_train)

preds = model.predict(x=X_train, batch_size=32)
```