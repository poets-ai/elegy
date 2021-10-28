import os

if "miniconda3/envs" in os.__file__:
    # specify the cuda location for XLA when working with conda environments
    os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=" + os.sep.join(
        os.__file__.split(os.sep)[:-3]
    )


import jax
import jax.numpy as jnp
from absl import app, flags

# importing tensorflow_datasets before performing any jax convolutions gives me a 'DNN Library not found' error later
# workaround: do a dummy convolution before importing tfds
_x0 = jnp.zeros((1, 1, 1, 1))
_x1 = jnp.zeros((1, 1, 1, 1))
jax.lax.conv(_x0, _x1, (1, 1), "SAME").block_until_ready()


import input_pipeline
import optax
import tensorflow_datasets as tfds

import elegy

print("JAX version:", jax.__version__)
print("Elegy version:", elegy.__version__)


FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "model",
    default=None,
    enum_values=[
        "ResNet18",
        "ResNet34",
        "ResNet50",
        "ResNet101",
        "ResNet152",
        "ResNet200",
    ],
    help="Type of ResNet to train",
)

flags.DEFINE_string(
    "output_dir",
    default=None,
    help="Directory to save model checkpoints and tensorboard log data",
)
flags.DEFINE_integer("epochs", default=90, help="Number of epochs to train")
flags.DEFINE_integer("batch_size", default=64, help="Input batch size")
flags.DEFINE_integer("image_size", default=224, help="Image size in pixels")
flags.DEFINE_string(
    "dataset", default="imagenet2012:*.*.*", help="TFDS dataset name and version"
)
flags.DEFINE_enum(
    "dtype",
    default="float32",
    enum_values=["float16", "float32"],
    help="Mixed precision or normal mode",
)
flags.DEFINE_float("base_lr", default=0.1, help="SGD optimizer base learning rate")
flags.DEFINE_float("momentum", default=0.9, help="SGD optimizer momentum")
flags.DEFINE_bool("nesterov", default=True, help="SGD optimizer Nesterov mode")
flags.DEFINE_float("L2_reg", default=1e-4, help="L2 weight regularization")
flags.DEFINE_bool("cache", default=False, help="Whether to cache the data in RAM")
flags.DEFINE_float(
    "loss_scale",
    default=1.0,
    help="Loss scale for numerical stability when dtype=float16",
)

flags.mark_flag_as_required("model")
flags.mark_flag_as_required("output_dir")


def main(argv):
    assert (
        len(argv) == 1
    ), "Please specify arguments via flags. Use --help for instructions"

    assert (
        getattr(elegy.nets.resnet, FLAGS.model, None) is not None
    ), f"{FLAGS.model} is not defined in elegy.nets.resnet"

    assert not os.path.exists(
        FLAGS.output_dir
    ), "Output directory already exists. Delete manually or specify a new one."
    os.makedirs(FLAGS.output_dir)

    # dataset
    dataset_builder = tfds.builder(FLAGS.dataset)
    ds_train = input_pipeline.create_split(
        dataset_builder,
        batch_size=FLAGS.batch_size,
        image_size=FLAGS.image_size,
        dtype=FLAGS.dtype,
        train=True,
        cache=FLAGS.cache,
    )
    ds_valid = input_pipeline.create_split(
        dataset_builder,
        batch_size=FLAGS.batch_size,
        image_size=FLAGS.image_size,
        dtype=FLAGS.dtype,
        train=False,
        cache=FLAGS.cache,
    )
    N_BATCHES_TRAIN = (
        dataset_builder.info.splits["train"].num_examples // FLAGS.batch_size
    )
    N_BATCHES_VALID = (
        dataset_builder.info.splits["validation"].num_examples // FLAGS.batch_size
    )

    # generator that converts tfds dataset batches to jax arrays
    def tfds2jax_generator(tf_ds):
        for batch in tf_ds:
            yield jnp.asarray(batch["image"], dtype=FLAGS.dtype), jax.device_put(
                jnp.asarray(batch["label"])
            )

    # model and optimizer definition
    def build_optimizer(
        lr, momentum, steps_per_epoch, n_epochs, nesterov, warmup_epochs=5
    ):
        cosine_schedule = optax.cosine_decay_schedule(
            1, decay_steps=n_epochs * steps_per_epoch, alpha=1e-10
        )
        warmup_schedule = optax.polynomial_schedule(
            init_value=0.0,
            end_value=1.0,
            power=1,
            transition_steps=warmup_epochs * steps_per_epoch,
        )
        schedule = lambda x: jnp.minimum(cosine_schedule(x), warmup_schedule(x))
        optimizer = optax.sgd(lr, momentum, nesterov=nesterov)
        optimizer = optax.chain(optimizer, optax.scale_by_schedule(schedule))
        return optimizer

    module = getattr(elegy.nets.resnet, FLAGS.model)(dtype=FLAGS.dtype)
    model = elegy.Model(
        module,
        loss=[
            elegy.losses.Crossentropy(from_logits=True, weight=FLAGS.loss_scale),
            elegy.regularizers.L2(FLAGS.L2_reg / 2 * FLAGS.loss_scale),
        ],
        metrics=elegy.metrics.Accuracy(),
        optimizer=build_optimizer(
            FLAGS.base_lr / FLAGS.loss_scale,
            FLAGS.momentum,
            N_BATCHES_TRAIN,
            FLAGS.epochs,
            FLAGS.nesterov,
        ),
    )

    # training
    model.fit(
        inputs=tfds2jax_generator(ds_train),
        validation_data=tfds2jax_generator(ds_valid),
        epochs=FLAGS.epochs,
        verbose=2,
        steps_per_epoch=N_BATCHES_TRAIN,
        validation_steps=N_BATCHES_VALID,
        callbacks=[
            elegy.callbacks.ModelCheckpoint(FLAGS.output_dir, save_best_only=True),
            elegy.callbacks.TerminateOnNaN(),
            elegy.callbacks.TensorBoard(logdir=FLAGS.output_dir),
        ],
    )


if __name__ == "__main__":
    app.run(main)
