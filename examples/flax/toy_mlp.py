# isort:skip_file
# fmt: off
import jax, optax
import numpy as np
import elegy as eg
import flax.linen as nn



# 1. create some data
x = np.random.uniform(-1, 1, size=(100, 1))
y = 1.3 * x ** 2 - 0.3 + 0.1 * np.random.normal(size=x.shape)



# 2. define the architecture
class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = jax.nn.relu(x)
        x = nn.Dense(1)(x)
        return x



# 3. create the Model
model = eg.Model(
    module=MLP(),
    loss=[
        eg.losses.MeanSquaredError(),
        eg.regularizers.L2(0.001),
    ],
    optimizer=optax.adam(1e-2),
)



# 4. train the model
model.fit(
    inputs=x,
    labels=y,
    epochs=100,
    callbacks=[eg.callbacks.TensorBoard("models/mlp/flax")],
)



# 5. visualize solution
import matplotlib.pyplot as plt

X_test = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
y_pred = model.predict(X_test)

plt.scatter(x, y)
plt.plot(X_test, y_pred)
plt.show()



# 6. save the model
model.save("models/mlp/flax/model")
model.saved_model(x, "models/mlp/flax/saved_model")
