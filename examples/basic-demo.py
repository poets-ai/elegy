import jax.numpy as jnp
import jax
import haiku as hk
import elegy
from jax.experimental import optix
import dataget
import matplotlib.pyplot as plt


class MLP(elegy.Module):
    """Standard LeNet-300-100 MLP network."""

    def __init__(self, n1: int = 300, n2: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.n1 = n1
        self.n2 = n2

    def call(self, image: jnp.ndarray) -> jnp.ndarray:
        image = image.astype(jnp.float32) / 255.0
        mlp = hk.Sequential(
            [
                hk.Flatten(),
                hk.Linear(self.n1),
                jax.nn.relu,
                hk.Linear(self.n2),
                jax.nn.relu,
                hk.Linear(10),
            ]
        )
        return mlp(image)


model = elegy.Model(
    module=lambda: MLP(n1=300, n2=100),
    loss=[
        elegy.losses.SparseCategoricalCrossentropy(from_logits=True),
        elegy.regularizers.GlobalL2(l=1e-4),
    ],
    metrics=lambda: elegy.metrics.SparseCategoricalAccuracy(),
    optimizer=optix.rmsprop(1e-3),
)

X_train, y_train, X_test, y_test = dataget.image.mnist(global_cache=True).get()
print("X_train:", X_train.shape, X_train.dtype)
print("y_train:", y_train.shape, y_train.dtype)
print("X_test:", X_test.shape, X_test.dtype)
print("y_test:", y_test.shape, y_test.dtype)
history = model.fit(
    x=X_train,
    y=y_train,
    epochs=100,
    steps_per_epoch=200,
    batch_size=64,
    validation_data=(X_test, y_test),
    shuffle=True,
)

def plot_history(history):
    n_plots = len(history.history.keys()) // 2
    plt.figure(figsize=(14, 24))

    for i, key in enumerate(list(history.history.keys())[:n_plots]):
        metric = history.history[key]
        val_metric = history.history[f"val_{key}"]

        plt.subplot(n_plots, 1, i + 1)
        plt.plot(metric, label=f"Training {key}")
        plt.plot(val_metric, label=f"Validation {key}")
        plt.legend(loc="lower right")
        plt.ylabel(key)
        plt.title(f"Training and Validation {key}")
    plt.show()


plot_history(history)