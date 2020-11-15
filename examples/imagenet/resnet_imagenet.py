#PARAMETERS
MODEL             = 'ResNet50'
OUTPUT_DIRECTORY  = 'models/resnet50'
EPOCHS            = 90
BATCH_SIZE        = 64
IMAGE_SIZE        = 224                                #image size in pixels
DATASET           = 'imagenet2012:5.0.*'               #TFDS dataset name and version
DTYPE             = 'float16'                          #float16 for mixed_precision or float32 for normal mode
LEARNING_RATE     = 0.1 * BATCH_SIZE / 256.
MOMENTUM          = 0.9
NESTEROV          = True
L2_REGULARIZATION = 1e-4
CACHE             = False                              #faster if True, but requires lots of RAM
LOSS_SCALE        = 256. if DTYPE=='float16' else 1.   #for numerical stability when DTYPE is float16



import os
if 'miniconda3/envs' in os.__file__:
    #specify the cuda location for XLA when working with conda environments
    os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir='+os.sep.join(os.__file__.split(os.sep)[:-3])


#importing tensorflow_datasets before performing any jax convolutions gives me a 'DNN Library not found' error later
#workaround: do a dummy convolution before importing tfds
import jax, jax.numpy as jnp
_x0 = jnp.zeros((1,1, 1,1))
_x1 = jnp.zeros((1,1, 1,1))
jax.lax.conv(_x0,_x1, (1,1), 'SAME').block_until_ready();


import elegy
import optax
import tensorflow_datasets as tfds
import input_pipeline

print('JAX version:',  jax.__version__)
print('Elegy version:',elegy.__version__)


assert getattr(elegy.nets.resnet, MODEL, None) is not None, f'{MODEL} is not defined in elegy.nets.resnet'

assert not os.path.exists(OUTPUT_DIRECTORY), 'Output directory already exists. Delete manually or specify a new one.'
os.makedirs(OUTPUT_DIRECTORY)


#dataset
dataset_builder = tfds.builder(DATASET)
ds_train = input_pipeline.create_split(dataset_builder,       batch_size=BATCH_SIZE, 
                                       image_size=IMAGE_SIZE, dtype=DTYPE, 
                                       train=True,            cache=CACHE)
ds_valid = input_pipeline.create_split(dataset_builder,       batch_size=BATCH_SIZE, 
                                       image_size=IMAGE_SIZE, dtype=DTYPE, 
                                       train=False,           cache=CACHE)
N_BATCHES_TRAIN = dataset_builder.info.splits['train'].num_examples // BATCH_SIZE
N_BATCHES_VALID = dataset_builder.info.splits['validation'].num_examples // BATCH_SIZE

#generator that converts tfds dataset batches to jax arrays
def tfds2jax_generator(tf_ds):
    for batch in tf_ds:
        yield jnp.asarray(batch['image'], dtype=DTYPE), jax.device_put(jnp.asarray(batch['label']))


#model and optimizer definition
def build_optimizer(lr, momentum, steps_per_epoch, n_epochs, warmup_epochs=5):
    cosine_schedule = optax.cosine_decay_schedule(1, decay_steps=n_epochs*steps_per_epoch, alpha=1e-10)
    warmup_schedule = optax.polynomial_schedule(init_value=0., end_value=1., power=1, transition_steps=warmup_epochs*steps_per_epoch)
    schedule        = lambda x: jnp.minimum(cosine_schedule(x), warmup_schedule(x))
    optimizer       = optax.sgd(lr, momentum, nesterov=NESTEROV)
    optimizer       = optax.chain(optimizer, optax.scale_by_schedule(schedule) )
    return optimizer

module = getattr(elegy.nets.resnet, MODEL)(dtype=DTYPE)
model  = elegy.Model(module,
                     loss=[
                           elegy.losses.SparseCategoricalCrossentropy(from_logits=True, weight=LOSS_SCALE),
                           elegy.regularizers.GlobalL2(L2_REGULARIZATION/2 * LOSS_SCALE)],
                     metrics=elegy.metrics.SparseCategoricalAccuracy(),
                     optimizer=build_optimizer(LEARNING_RATE / LOSS_SCALE, MOMENTUM, N_BATCHES_TRAIN, EPOCHS),
                      )


#training
model.fit( x                 = tfds2jax_generator(ds_train),
           validation_data   = tfds2jax_generator(ds_valid),
           epochs            = EPOCHS,
           verbose           = 2,
           steps_per_epoch   = N_BATCHES_TRAIN, 
           validation_steps  = N_BATCHES_VALID,
           callbacks         = [elegy.callbacks.ModelCheckpoint(OUTPUT_DIRECTORY, save_best_only=True),
                                elegy.callbacks.TerminateOnNaN(),
                                elegy.callbacks.TensorBoard(logdir=OUTPUT_DIRECTORY)]
         )
