class MLP(hk.Module):
    def __call__(self, image) -> jnp.ndarray:
        """Standard LeNet-300-100 MLP network."""
        image = image.astype(jnp.float32) / 255.0

        mlp = hk.Sequential(
            [
                hk.Flatten(),
                hk.Linear(300),
                jax.nn.relu,
                hk.Linear(100),
                jax.nn.relu,
                hk.Linear(10),
            ]
        )
        return mlp(image)


model = elegy.Model(
    module=lambda: MLP(),
    loss=lambda: elegy.losses.SparseCategoricalCrossentropy(from_logits=True),
    aux_losses=lambda: elegy.regularizers.GlobalL2(l=1e-5),
    metrics=lambda: elegy.metrics.SparseCategoricalAccuracy(),
    optimizer=optix.rmsprop(0.001),
)
