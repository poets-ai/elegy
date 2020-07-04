# Lint as: python3
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""MNIST classifier example."""

from typing import Any, Generator, Mapping, Tuple

import typer
import haiku as hk
import jax
from jax.experimental import optix
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds
import elegy

OptState = Any
Batch = Mapping[str, np.ndarray]


def net_fn(image) -> jnp.ndarray:
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


def load_dataset(
    split: str, *, is_training: bool, batch_size: int,
) -> Generator[Batch, None, None]:
    """Loads the dataset as a generator of batches."""
    ds = tfds.load("mnist:3.*.*", split=split).cache().repeat()
    if is_training:
        ds = ds.shuffle(10 * batch_size, seed=0)
    ds = ds.batch(batch_size)
    return tfds.as_numpy(ds)


def accuracy(y_true, y_pred):
    return jnp.mean(jnp.argmax(y_pred, axis=-1) == y_true)


def main(debug: bool = False, eager: bool = False):

    if debug:
        import debugpy

        print("Waiting for debugger...")
        debugpy.listen(5678)
        debugpy.wait_for_client()

    # Make the network and optimiser.
    # net = hk.transform(net_fn)
    # opt = optix.adam(1e-3)

    # Training loss (cross-entropy).
    def loss(y_true, y_pred, params) -> jnp.ndarray:
        """Compute the loss of the network, including L2."""
        # logits = net.apply(params, x)
        labels = jax.nn.one_hot(y_true, 10)

        l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))

        softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(y_pred))
        softmax_xent /= labels.shape[0]

        return softmax_xent + 1e-4 * l2_loss

    # Evaluation metric (classification accuracy).

    # @jax.jit
    # def update(
    #     params: hk.Params, opt_state: OptState, batch: Batch,
    # ) -> Tuple[jnp.array, hk.Params, OptState]:
    #     """Learning rule (stochastic gradient descent)."""
    #     loss_val, grads = jax.value_and_grad(loss)(params, batch)
    #     updates, opt_state = opt.update(grads, opt_state)
    #     new_params = optix.apply_updates(params, updates)
    #     return loss_val, new_params, opt_state

    # We maintain avg_params, the exponential moving average of the "live" params.
    # avg_params is used only for evaluation.
    # For more, see: https://doi.org/10.1137/0330046
    # @jax.jit
    # def ema_update(
    #     avg_params: hk.Params, new_params: hk.Params, epsilon: float = 0.001,
    # ) -> hk.Params:
    #     return jax.tree_multimap(
    #         lambda p1, p2: (1 - epsilon) * p1 + epsilon * p2, avg_params, new_params
    #     )

    # Make datasets.
    train = load_dataset("train", is_training=True, batch_size=64)
    train_eval = load_dataset("train", is_training=False, batch_size=1000)
    test_eval = load_dataset("test", is_training=False, batch_size=1000)

    # Initialize network and optimiser; note we draw an input to get shapes.
    # sample = next(train)
    # print(sample["image"].shape)
    # params = avg_params = net.init(jax.random.PRNGKey(42), sample)
    # opt_state = opt.init(params)
    loss_acc = 0
    logs = None

    model = elegy.Model(
        net_fn=net_fn,
        loss=loss,
        metrics=lambda: [("accuracy", accuracy)],
        run_eagerly=eager,
    )

    # Train/eval loop.
    for step in range(10001):
        if step > 0 and step % 1000 == 0:
            # Periodically evaluate classification accuracy on train & test sets.
            # train_accuracy = accuracy(avg_params, next(train_eval))
            # test_accuracy = accuracy(avg_params, next(test_eval))
            # train_accuracy, test_accuracy = jax.device_get(
            #     (train_accuracy, test_accuracy)
            # )
            print(
                f"[Step {step}] Train / Test accuracy: "
                f"{logs['accuracy']} - "
                f"Train Loss: {loss_acc/1000:.3f}"
            )
            loss_acc = 0

        # Do SGD on a batch of training examples.
        sample = next(train)

        logs, _0, _1, _2, _3 = model.train_on_batch(x=sample, y=sample["label"])

        # loss_val, params, opt_state = update(params, opt_state, sample)
        # avg_params = ema_update(avg_params, params)

        loss_acc += logs["loss"]

        # print(step, loss_val)


if __name__ == "__main__":
    typer.run(main)
