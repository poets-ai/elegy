# test_step
The `test_step` computes the main `loss` of the model along with some `logs` for reporting, by overriding this method you can directly influence what happens during `evaluate`.

### Inputs
Any of following input arguments are available for `test_step`:

| name            | type                |                                             |
| :-------------- | :------------------ | :------------------------------------------ |
| `x`             | `Any`               | Input data                                  |
| `y_true`        | `Any`               | The target labels                           |
| `sample_weight` | `Optional[ndarray]` | The weight of each sample in the total loss |
| `class_weight`  | `Optional[ndarray]` | The weight of each class in the total loss  |
| `states`        | `States`            | Current state of the model                  |
| `initializing`  | `bool`              | Whether the model is initializing or not    |
| `training`      | `bool`              | Whether the model is training or not        |


You must request the arguments you want by **name**.

### Outputs
`pred_step` must output a tuple with the following values:

| name     | type                 |                                             |
| :------- | :------------------- | :------------------------------------------ |
| `loss`   | `ndarray`            | The loss of the model over the data         |
| `logs`   | `Dict[str, ndarray]` | A dictionary with a set of values to report |
| `states` | `States`             | The new state of the model                  |


### Callers
| method       | when                                              |
| :----------- | :------------------------------------------------ |
| `evaluate`   | always                                            |
| `grad_step`  | default implementation                            |
| `train_step` | default implementation during initialization only |

### Examples
Lets review the example of `test_step` found in [basics](./basics):

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
In this case `test_step` is defining both the "forward" pass of the model and calculating the losses and metrics in a single place. However, since we are not defining `pred_step` we loose the power to call `predict` which might not be desirable. The optimimal way to fix this is to extract the calculation of the logits into `pred_step` and call this from `test_step`:

```python
class LinearClassifier(elegy.Model):
    def test_step(self, x, states, initializing):  
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

        return logits, states.update(net_params=(w, b))

    def test_step(self, x, y_true, states, initializing):  
        # call pred_step
        logits, states = self.pred_step((x, states, initializing)

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
    optimizer=optax.adam(1e-3),
)

model.fit(
    x=X_train,
    y=y_train,
    epochs=100,
    batch_size=64,
)
```
This not only creates a separation of concerns, it also favors code reuse, and we can now use `predict`, `evaluate`, and `fit` as intended. 

There are cases however where you might want to implement a forward pass inside `test_step` that is different from what you would define in `pred_step`, for example you can create a `VAE` or `GAN` Models that use multiple modules to calculate the loss inside `test_step` (e.g. encoder, decoder, and discriminator) but only use the decoder inside `pred_step` to generate samples.

### Default Implementation
The default implementation of `pred_step` does the following:
* Call `pred_step` to get `y_pred`. 
* Calls `api_loss.init` or `api_loss.apply` depending on state of `initializing`. `api_loss` of type `Losses` computes the aggregated batch loss from the loss functions passed by the user through the `loss` argument in the `Model`s constructor, and also computes a running mean of each loss individually which is passed for reporting to `logs`.
* Calls `api_metrics.init` or `api_metrics.apply` depending on state of `initializing`. `api_metrics` of type `Metrics` calculates the metrics passed by the user through the `metrics` argument in the `Model`s constructor and passes their values to `logs` for reporting.