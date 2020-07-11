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


def model_fn(x) -> jnp.ndarray:
    """Standard LeNet-300-100 MLP network."""
    x = x.astype(jnp.float32) / 255.0

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
    return mlp(x)


def load_dataset(
    split: str, *, is_training: bool, batch_size: int,
) -> Generator[Batch, None, None]:
    """Loads the dataset as a generator of batches."""
    ds = tfds.load("mnist:3.*.*", split=split).cache().repeat()
    if is_training:
        ds = ds.shuffle(10 * batch_size, seed=0)
    ds = ds.batch(batch_size)
    return tfds.as_numpy(ds)


def main(debug: bool = False):

    # Make datasets.
    train = load_dataset("train", is_training=True, batch_size=64)
    train_eval = load_dataset("train", is_training=False, batch_size=1000)
    test_eval = load_dataset("test", is_training=False, batch_size=1000)

    model = elegy.Model(
        model_fn=model_fn,
        loss=SoftmaxCrossEntropy(),
        metrics=[Accuracy()],
        run_eagerly=False,
    )

    model.fit()


if __name__ == "__main__":
    typer.run(main)
