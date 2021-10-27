import os
from datetime import datetime

import dataget
import elegy
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import torch
import typer
from elegy.callbacks.tensorboard import TensorBoard
from tensorboardX.writer import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset


def main(
    debug: bool = False,
    eager: bool = False,
    logdir: str = "runs",
    steps_per_epoch: int = 200,
    epochs: int = 100,
    batch_size: int = 64,
):

    if debug:
        import debugpy

        print("Waiting for debugger...")
        debugpy.listen(5678)
        debugpy.wait_for_client()

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    logdir = os.path.join(logdir, current_time)

    X_train, y_train, X_test, y_test = dataget.image.mnist(global_cache=True).get()

    X_train = X_train[..., None]
    X_test = X_test[..., None]

    print("X_train:", X_train.shape, X_train.dtype)
    print("y_train:", y_train.shape, y_train.dtype)
    print("X_test:", X_test.shape, X_test.dtype)
    print("y_test:", y_test.shape, y_test.dtype)

    class CNN(elegy.Module):
        def call(self, image: jnp.ndarray, training: bool):
            @elegy.to_module
            def ConvBlock(x, units, kernel, stride=1):
                x = elegy.nn.Conv2D(units, kernel, stride=stride, padding="same")(x)
                x = elegy.nn.BatchNormalization()(x, training)
                x = elegy.nn.Dropout(0.2)(x, training)
                return jax.nn.relu(x)

            x: np.ndarray = image.astype(jnp.float32) / 255.0

            # base
            x = ConvBlock()(x, 32, [3, 3])
            x = ConvBlock()(x, 64, [3, 3], stride=2)
            x = ConvBlock()(x, 64, [3, 3], stride=2)
            x = ConvBlock()(x, 128, [3, 3], stride=2)

            # GlobalAveragePooling2D
            x = jnp.mean(x, axis=[1, 2])

            # 1x1 Conv
            x = elegy.nn.Linear(10)(x)

            return x

    model = elegy.Model(
        module=CNN(),
        loss=elegy.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=elegy.metrics.Accuracy(),
        optimizer=optax.adam(1e-3),
        eager=eager,
    )

    # show summary
    model.summary(X_train[:64])

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    history = model.fit(
        train_dataloader,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_dataloader,
        callbacks=[TensorBoard(logdir=logdir)],
    )

    elegy.utils.plot_history(history)

    model.save("models/conv")

    model = elegy.load("models/conv")

    print(model.evaluate(x=X_test, y=y_test))

    # get random samples
    idxs = np.random.randint(0, 10000, size=(9,))
    x_sample = X_test[idxs]

    # get predictions
    y_pred = model.predict(x=x_sample)

    # plot results
    with SummaryWriter(os.path.join(logdir, "val")) as tbwriter:
        figure = plt.figure(figsize=(12, 12))
        for i in range(3):
            for j in range(3):
                k = 3 * i + j
                plt.subplot(3, 3, k + 1)

                plt.title(f"{np.argmax(y_pred[k])}")
                plt.imshow(x_sample[k], cmap="gray")
        # tbwriter.add_figure("Conv classifier", figure, 100)

    plt.show()


if __name__ == "__main__":
    typer.run(main)
