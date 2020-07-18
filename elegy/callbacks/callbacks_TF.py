# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=g-import-not-at-top
"""Callbacks: utilities called at certain points during model training.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import csv
import io
import json
import os
import re
import tempfile
import time

import numpy as np
import six

# from tensorflow.python.data.ops import iterator_ops

# from tensorflow.python.distribute import distributed_file_utils
# from tensorflow.python.distribute import multi_worker_util
# from tensorflow.python.eager import context
# from tensorflow.python.framework import ops
# from tensorflow.python.keras import backend as K
# from tensorflow.python.keras.distribute import (
#     multi_worker_training_state as training_state,
# )
# from tensorflow.python.keras.utils import
# from tensorflow.python.keras.utils import tf_utils
# from tensorflow.python.keras.utils.data_utils import Sequence
# from tensorflow.python.keras.utils.generic_utils import Progbar

import sys


# from tensorflow.python.ops import array_ops
# from tensorflow.python.ops import control_flow_ops
# from tensorflow.python.ops import math_ops
# from tensorflow.python.ops import summary_ops_v2
# from tensorflow.python.ops import variables
# from tensorflow.python.platform import tf_logging as logging
# from tensorflow.python.profiler import profiler_v2 as profiler
# from tensorflow.python.training import checkpoint_management
# from tensorflow.python.util import nest
# from tensorflow.python.util.compat import collections_abc
# from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls

# try:
#     import requests
# except ImportError:
#     requests = None

import logging

import typing as tp


# class TensorBoard(Callback):
#     # pylint: disable=line-too-long
#     """Enable visualizations for TensorBoard.

#   TensorBoard is a visualization tool provided with TensorFlow.

#   This callback logs events for TensorBoard, including:

#   * Metrics summary plots
#   * Training graph visualization
#   * Activation histograms
#   * Sampled profiling

#   If you have installed TensorFlow with pip, you should be able
#   to launch TensorBoard from the command line:

#   ```sh
#   tensorboard --logdir=path_to_your_logs
#   ```

#   You can find more information about TensorBoard
#   [here](https://www.tensorflow.org/get_started/summaries_and_tensorboard).

#   Example (Basic):
#   ```python
#   tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
#   model.fit(x_train, y_train, epochs=2, callbacks=[tensorboard_callback])
#   # run the tensorboard command to view the visualizations.
#   ```
#   Example (Profile):
#   ```python
#   # profile a single batch, e.g. the 5th batch.
#   tensorboard_callback =
#       tf.keras.callbacks.TensorBoard(log_dir='./logs', profile_batch=5)
#   model.fit(x_train, y_train, epochs=2, callbacks=[tensorboard_callback])
#   # run the tensorboard command to view the visualizations in profile plugin.

#   # profile a range of batches, e.g. from 10 to 20.
#   tensorboard_callback =
#       tf.keras.callbacks.TensorBoard(log_dir='./logs', profile_batch='10,20')
#   model.fit(x_train, y_train, epochs=2, callbacks=[tensorboard_callback])
#   # run the tensorboard command to view the visualizations in profile plugin.
#   ```

#   Arguments:
#       log_dir: the path of the directory where to save the log files to be
#         parsed by TensorBoard.
#       histogram_freq: frequency (in epochs) at which to compute activation and
#         weight histograms for the layers of the model. If set to 0, histograms
#         won't be computed. Validation data (or split) must be specified for
#         histogram visualizations.
#       write_graph: whether to visualize the graph in TensorBoard. The log file
#         can become quite large when write_graph is set to True.
#       write_images: whether to write model weights to visualize as image in
#         TensorBoard.
#       update_freq: `'batch'` or `'epoch'` or integer. When using `'batch'`,
#         writes the losses and metrics to TensorBoard after each batch. The same
#         applies for `'epoch'`. If using an integer, let's say `1000`, the
#         callback will write the metrics and losses to TensorBoard every 1000
#         batches. Note that writing too frequently to TensorBoard can slow down
#         your training.
#       profile_batch: Profile the batch(es) to sample compute characteristics.
#         profile_batch must be a non-negative integer or a comma separated string
#         of pair of positive integers. A pair of positive integers signify a
#         range of batches to profile. By default, it will profile the second
#         batch. Set profile_batch=0 to disable profiling. Must run in TensorFlow
#         eager mode.
#       embeddings_freq: frequency (in epochs) at which embedding layers will be
#         visualized. If set to 0, embeddings won't be visualized.
#       embeddings_metadata: a dictionary which maps layer name to a file name in
#         which metadata for this embedding layer is saved. See the
#         [details](
#           https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
#         about metadata files format. In case if the same metadata file is
#         used for all embedding layers, string can be passed.

#   Raises:
#       ValueError: If histogram_freq is set and no validation data is provided.
#   """

#     # pylint: enable=line-too-long

#     def __init__(
#         self,
#         log_dir="logs",
#         histogram_freq=0,
#         write_graph=True,
#         write_images=False,
#         update_freq="epoch",
#         profile_batch=2,
#         embeddings_freq=0,
#         embeddings_metadata=None,
#         **kwargs
#     ):
#         super(TensorBoard, self).__init__()
#         self._validate_kwargs(kwargs)

#         self.log_dir = log_dir
#         self.histogram_freq = histogram_freq
#         self.write_graph = write_graph
#         self.write_images = write_images
#         if update_freq == "batch":
#             self.update_freq = 1
#         else:
#             self.update_freq = update_freq
#         self.embeddings_freq = embeddings_freq
#         self.embeddings_metadata = embeddings_metadata

#         self._samples_seen = 0
#         self._samples_seen_at_last_write = 0
#         self._current_batch = 0

#         # A collection of file writers currently in use, to be closed when
#         # training ends for this callback. Writers are keyed by the
#         # directory name under the root logdir: e.g., "train" or
#         # "validation".
#         self._train_run_name = "train"
#         self._validation_run_name = "validation"
#         self._writers = {}
#         self._start_batch, self._stop_batch = self._init_profile_batch(profile_batch)
#         if self._start_batch > 0:
#             profiler.warmup()  # Improve the profiling accuracy.
#         # True when a trace is running.
#         self._is_tracing = False

#     def _validate_kwargs(self, kwargs):
#         """Handle arguments were supported in V1."""
#         if kwargs.get("write_grads", False):
#             logging.warning(
#                 "`write_grads` will be ignored in TensorFlow 2.0 "
#                 "for the `TensorBoard` Callback."
#             )
#         if kwargs.get("batch_size", False):
#             logging.warning(
#                 "`batch_size` is no longer needed in the "
#                 "`TensorBoard` Callback and will be ignored "
#                 "in TensorFlow 2.0."
#             )
#         if kwargs.get("embeddings_layer_names", False):
#             logging.warning(
#                 "`embeddings_layer_names` is not supported in "
#                 "TensorFlow 2.0. Instead, all `Embedding` layers "
#                 "will be visualized."
#             )
#         if kwargs.get("embeddings_data", False):
#             logging.warning(
#                 "`embeddings_data` is not supported in TensorFlow "
#                 "2.0. Instead, all `Embedding` variables will be "
#                 "visualized."
#             )

#         unrecognized_kwargs = set(kwargs.keys()) - {
#             "write_grads",
#             "embeddings_layer_names",
#             "embeddings_data",
#             "batch_size",
#         }

#         # Only allow kwargs that were supported in V1.
#         if unrecognized_kwargs:
#             raise ValueError(
#                 "Unrecognized arguments in `TensorBoard` "
#                 "Callback: " + str(unrecognized_kwargs)
#             )

#     def set_model(self, model):
#         """Sets Keras model and writes graph if specified."""
#         self.model = model

#         # In case this callback is used via native Keras, _get_distribution_strategy does not exist.
#         if hasattr(self.model, "_get_distribution_strategy"):
#             # TensorBoard callback involves writing a summary file in a
#             # possibly distributed settings.
#             self._log_write_dir = distributed_file_utils.write_dirpath(
#                 self.log_dir, self.model._get_distribution_strategy()
#             )  # pylint: disable=protected-access
#         else:
#             self._log_write_dir = self.log_dir

#         with context.eager_mode():
#             self._close_writers()
#             if self.write_graph:
#                 with self._get_writer(self._train_run_name).as_default():
#                     with summary_ops_v2.always_record_summaries():
#                         if not model.run_eagerly:
#                             summary_ops_v2.graph(K.get_graph(), step=0)

#                         summary_writable = (
#                             self.model._is_graph_network
#                             or self.model.__class__.__name__  # pylint: disable=protected-access
#                             == "Sequential"
#                         )  # pylint: disable=protected-access
#                         if summary_writable:
#                             summary_ops_v2.keras_model("keras", self.model, step=0)

#         if self.embeddings_freq:
#             self._configure_embeddings()

#         summary_state = (
#             summary_ops_v2._summary_state
#         )  # pylint: disable=protected-access
#         self._prev_summary_recording = summary_state.is_recording
#         self._prev_summary_writer = summary_state.writer
#         self._prev_summary_step = summary_state.step

#     def _configure_embeddings(self):
#         """Configure the Projector for embeddings."""
#         # TODO(omalleyt): Add integration tests.
#         from tensorflow.python.keras.layers import embeddings

#         try:
#             from tensorboard.plugins import projector
#         except ImportError:
#             raise ImportError(
#                 "Failed to import TensorBoard. Please make sure that "
#                 'TensorBoard integration is complete."'
#             )
#         config = projector.ProjectorConfig()
#         for layer in self.model.layers:
#             if isinstance(layer, embeddings.Embedding):
#                 embedding = config.embeddings.add()
#                 embedding.tensor_name = layer.embeddings.name

#                 if self.embeddings_metadata is not None:
#                     if isinstance(self.embeddings_metadata, str):
#                         embedding.metadata_path = self.embeddings_metadata
#                     else:
#                         if layer.name in embedding.metadata_path:
#                             embedding.metadata_path = self.embeddings_metadata.pop(
#                                 layer.name
#                             )

#         if self.embeddings_metadata:
#             raise ValueError(
#                 "Unrecognized `Embedding` layer names passed to "
#                 "`keras.callbacks.TensorBoard` `embeddings_metadata` "
#                 "argument: " + str(self.embeddings_metadata.keys())
#             )

#         class DummyWriter(object):
#             """Dummy writer to conform to `Projector` API."""

#             def __init__(self, logdir):
#                 self.logdir = logdir

#             def get_logdir(self):
#                 return self.logdir

#         writer = DummyWriter(self._log_write_dir)
#         projector.visualize_embeddings(writer, config)

#     def _close_writers(self):
#         """Close all remaining open file writers owned by this callback.

#     If there are no such file writers, this is a no-op.
#     """
#         with context.eager_mode():
#             for writer in six.itervalues(self._writers):
#                 writer.close()
#             self._writers.clear()

#     def _get_writer(self, writer_name):
#         """Get a summary writer for the given subdirectory under the logdir.

#     A writer will be created if it does not yet exist.

#     Arguments:
#       writer_name: The name of the directory for which to create or
#         retrieve a writer. Should be either `self._train_run_name` or
#         `self._validation_run_name`.

#     Returns:
#       A `SummaryWriter` object.
#     """
#         if writer_name not in self._writers:
#             path = os.path.join(self._log_write_dir, writer_name)
#             writer = summary_ops_v2.create_file_writer_v2(path)
#             self._writers[writer_name] = writer
#         return self._writers[writer_name]

#     def _set_default_writer(self, writer_name):
#         """Sets the default writer for custom batch-level summaries."""
#         if self.update_freq == "epoch":
#             # Writer is only used for custom summaries, which are written
#             # batch-by-batch.
#             return

#         step = self._total_batches_seen[writer_name]

#         def _should_record():
#             return math_ops.equal(step % self.update_freq, 0)

#         summary_state = (
#             summary_ops_v2._summary_state
#         )  # pylint: disable=protected-access
#         summary_state.is_recording = _should_record
#         summary_state.writer = self._get_writer(writer_name)
#         summary_ops_v2.set_step(step)

#     def _init_batch_steps(self):
#         """Create the total batch counters."""
#         if ops.executing_eagerly_outside_functions():
#             # Variables are needed for the `step` value of custom tf.summaries
#             # to be updated inside a tf.function.
#             self._total_batches_seen = {
#                 self._train_run_name: variables.Variable(0, dtype="int64"),
#                 self._validation_run_name: variables.Variable(0, dtype="int64"),
#             }
#         else:
#             # Custom tf.summaries are not supported in legacy graph mode.
#             self._total_batches_seen = {
#                 self._train_run_name: 0,
#                 self._validation_run_name: 0,
#             }

#     def _increment_step(self, writer_name):
#         step = self._total_batches_seen[writer_name]
#         if isinstance(step, variables.Variable):
#             step.assign_add(1)
#         else:
#             self._total_batches_seen[writer_name] += 1

#     def _init_profile_batch(self, profile_batch):
#         """Validate profile_batch value and set the range of batches to profile.

#     Arguments:
#       profile_batch: The range of batches to profile. Should be a non-negative
#         integer or a comma separated string of pair of positive integers. A pair
#         of positive integers signify a range of batches to profile.

#     Returns:
#       A pair of non-negative integers specifying the start and stop batch to
#       profile.

#     Raises:
#       ValueError: If profile_batch is not an integer or a comma seperated pair
#                   of positive integers.

#     """
#         profile_batch_error_message = (
#             "profile_batch must be a non-negative integer or a comma separated "
#             "string of pair of positive integers. A pair of positive integers "
#             "signify a range of batches to profile."
#         )
#         try:
#             profile_range = [int(i) for i in str(profile_batch).split(",")]
#         except ValueError:
#             raise ValueError(profile_batch_error_message)
#         if len(profile_range) == 1:  # single batch
#             start_batch, stop_batch = profile_range[0], profile_range[0]
#             if start_batch < 0:
#                 raise ValueError(profile_batch_error_message)
#         elif len(profile_range) == 2:  # (start_batch, stop_batch)
#             start_batch, stop_batch = profile_range
#             # [0, 0], [-1, 100], [6, 5] are illegal.
#             if start_batch <= 0 or start_batch > stop_batch:
#                 raise ValueError(profile_batch_error_message)
#         else:
#             raise ValueError(profile_batch_error_message)
#         return start_batch, stop_batch

#     def on_train_begin(self, logs=None):
#         self._init_batch_steps()
#         if self._start_batch == 1:
#             self._enable_trace()

#     def on_test_begin(self, logs=None):
#         self._set_default_writer(self._validation_run_name)

#     def on_train_batch_end(self, batch, logs=None):
#         """Writes scalar summaries for metrics on every training batch.

#     Performs profiling if current batch is in profiler_batches.

#     Arguments:
#       batch: Integer, index of batch within the current epoch.
#       logs: Dict. Metric results for this batch.
#     """
#         # TODO(b/150629188): Make TensorBoard callback not use batch hooks
#         # by default.
#         if self.update_freq == "epoch" and self._start_batch is None:
#             return

#         # Don't output batch_size and batch number as TensorBoard summaries
#         logs = logs or {}
#         train_batches = self._total_batches_seen[self._train_run_name]
#         if self.update_freq != "epoch" and batch % self.update_freq == 0:
#             self._log_metrics(logs, prefix="batch_", step=train_batches)

#         self._increment_step(self._train_run_name)
#         if self._is_tracing:
#             control_flow_ops.cond(
#                 math_ops.greater_equal(train_batches, self._stop_batch),
#                 lambda: self._log_trace_return_true(),
#                 lambda: False,
#             )  # pylint: disable=unnecessary-lambda
#         else:
#             control_flow_ops.cond(
#                 math_ops.equal(train_batches, self._start_batch - 1),
#                 lambda: self._enable_trace_return_true(),
#                 lambda: False,
#             )  # pylint: disable=unnecessary-lambda

#     def on_test_batch_end(self, batch, logs=None):
#         if self.update_freq == "epoch":
#             return
#         self._increment_step(self._validation_run_name)

#     def on_epoch_begin(self, epoch, logs=None):
#         self._set_default_writer(self._train_run_name)

#     def on_epoch_end(self, epoch, logs=None):
#         """Runs metrics and histogram summaries at epoch end."""
#         self._log_metrics(logs, prefix="epoch_", step=epoch)

#         if self.histogram_freq and epoch % self.histogram_freq == 0:
#             self._log_weights(epoch)

#         if self.embeddings_freq and epoch % self.embeddings_freq == 0:
#             self._log_embeddings(epoch)

#     def on_train_end(self, logs=None):
#         if self._is_tracing:
#             self._log_trace()
#         self._close_writers()

#         summary_state = (
#             summary_ops_v2._summary_state
#         )  # pylint: disable=protected-access
#         summary_state.is_recording = self._prev_summary_recording
#         summary_state.writer = self._prev_summary_writer
#         summary_state.step = self._prev_summary_step

#         # In case this callback is used via native Keras, _get_distribution_strategy does not exist.
#         if hasattr(self.model, "_get_distribution_strategy"):
#             # Safely remove the unneeded temp files.
#             distributed_file_utils.remove_temp_dirpath(
#                 self.log_dir, self.model._get_distribution_strategy()
#             )  # pylint: disable=protected-access

#     def _enable_trace(self):
#         """Starts to collect trace graph to TensorBoard.

#     Collects both trace and graph in eager mode, and trace only in graph mode.
#     """
#         if context.executing_eagerly():
#             # Graph must be traced in eager mode.
#             summary_ops_v2.trace_on(graph=True, profiler=False)
#         profiler.start(logdir=os.path.join(self._log_write_dir, "train"))
#         self._is_tracing = True

#     def _enable_trace_return_true(self):
#         """Starts to collect trace graph to TensorBoard and returns True.

#     Returns:
#       True.
#     """
#         self._enable_trace()
#         return True

#     def _log_trace(self):
#         """Logs the trace graph to TensorBoard.

#     Logs both trace and graph in eager mode, and trace only in graph mode.
#     """
#         profiler.stop()
#         if context.executing_eagerly():
#             # Graph must be traced in eager mode.
#             with self._get_writer(
#                 self._train_run_name
#             ).as_default(), summary_ops_v2.always_record_summaries():
#                 # TODO(b/126388999): Remove step info in the summary name.
#                 step = K.get_value(self._total_batches_seen[self._train_run_name])
#                 summary_ops_v2.trace_export(name="batch_%d" % step, step=step)
#         self._is_tracing = False

#     def _log_trace_return_true(self):
#         """Logs the trace graph to TensorBoard and returns True.

#     Returns:
#       True.
#     """
#         self._log_trace()
#         return True

#     def _log_metrics(self, logs, prefix, step):
#         """Writes metrics out as custom scalar summaries.

#     Arguments:
#         logs: Dict. Keys are scalar summary names, values are NumPy scalars.
#         prefix: String. The prefix to apply to the scalar summary names.
#         step: Int. The global step to use for TensorBoard.
#     """
#         if logs is None:
#             logs = {}

#         # Group metrics by the name of their associated file writer. Values
#         # are lists of metrics, as (name, scalar_value) pairs.
#         logs_by_writer = {
#             self._train_run_name: [],
#             self._validation_run_name: [],
#         }
#         validation_prefix = "val_"
#         for (name, value) in logs.items():
#             if name in ("batch", "size", "num_steps"):
#                 # Scrub non-metric items.
#                 continue
#             if name.startswith(validation_prefix):
#                 name = name[len(validation_prefix) :]
#                 writer_name = self._validation_run_name
#             else:
#                 writer_name = self._train_run_name
#             name = prefix + name  # assign batch or epoch prefix
#             logs_by_writer[writer_name].append((name, value))

#         with context.eager_mode():
#             with summary_ops_v2.always_record_summaries():
#                 for writer_name in logs_by_writer:
#                     these_logs = logs_by_writer[writer_name]
#                     if not these_logs:
#                         # Don't create a "validation" events file if we don't
#                         # actually have any validation data.
#                         continue
#                     writer = self._get_writer(writer_name)
#                     with writer.as_default():
#                         for (name, value) in these_logs:
#                             summary_ops_v2.scalar(name, value, step=step)

#     def _log_weights(self, epoch):
#         """Logs the weights of the Model to TensorBoard."""
#         writer = self._get_writer(self._train_run_name)
#         with context.eager_mode(), writer.as_default(), summary_ops_v2.always_record_summaries():
#             for layer in self.model.layers:
#                 for weight in layer.weights:
#                     weight_name = weight.name.replace(":", "_")
#                     with ops.init_scope():
#                         weight = K.get_value(weight)
#                     summary_ops_v2.histogram(weight_name, weight, step=epoch)
#                     if self.write_images:
#                         self._log_weight_as_image(weight, weight_name, epoch)
#             writer.flush()

#     def _log_weight_as_image(self, weight, weight_name, epoch):
#         """Logs a weight as a TensorBoard image."""
#         w_img = array_ops.squeeze(weight)
#         shape = K.int_shape(w_img)
#         if len(shape) == 1:  # Bias case
#             w_img = array_ops.reshape(w_img, [1, shape[0], 1, 1])
#         elif len(shape) == 2:  # Dense layer kernel case
#             if shape[0] > shape[1]:
#                 w_img = array_ops.transpose(w_img)
#                 shape = K.int_shape(w_img)
#             w_img = array_ops.reshape(w_img, [1, shape[0], shape[1], 1])
#         elif len(shape) == 3:  # ConvNet case
#             if K.image_data_format() == "channels_last":
#                 # Switch to channels_first to display every kernel as a separate
#                 # image.
#                 w_img = array_ops.transpose(w_img, perm=[2, 0, 1])
#                 shape = K.int_shape(w_img)
#             w_img = array_ops.reshape(w_img, [shape[0], shape[1], shape[2], 1])

#         shape = K.int_shape(w_img)
#         # Not possible to handle 3D convnets etc.
#         if len(shape) == 4 and shape[-1] in [1, 3, 4]:
#             summary_ops_v2.image(weight_name, w_img, step=epoch)

#     def _log_embeddings(self, epoch):
#         embeddings_ckpt = os.path.join(
#             self._log_write_dir, "train", "keras_embedding.ckpt-{}".format(epoch)
#         )
#         self.model.save_weights(embeddings_ckpt)


# @keras_export("keras.callbacks.LearningRateScheduler")
# class LearningRateScheduler(Callback):
#     """Learning rate scheduler.

#   Arguments:
#       schedule: a function that takes an epoch index as input
#           (integer, indexed from 0) and returns a new
#           learning rate as output (float).
#       verbose: int. 0: quiet, 1: update messages.

#   ```python
#   # This function keeps the learning rate at 0.001 for the first ten epochs
#   # and decreases it exponentially after that.
#   def scheduler(epoch):
#     if epoch < 10:
#       return 0.001
#     else:
#       return 0.001 * tf.math.exp(0.1 * (10 - epoch))

#   callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
#   model.fit(data, labels, epochs=100, callbacks=[callback],
#             validation_data=(val_data, val_labels))
#   ```
#   """

#     def __init__(self, schedule, verbose=0):
#         super(LearningRateScheduler, self).__init__()
#         self.schedule = schedule
#         self.verbose = verbose

#     def on_epoch_begin(self, epoch, logs=None):
#         if not hasattr(self.model.optimizer, "lr"):
#             raise ValueError('Optimizer must have a "lr" attribute.')
#         try:  # new API
#             lr = float(K.get_value(self.model.optimizer.lr))
#             lr = self.schedule(epoch, lr)
#         except TypeError:  # Support for old API for backward compatibility
#             lr = self.schedule(epoch)
#         if not isinstance(lr, (ops.Tensor, float, np.float32, np.float64)):
#             raise ValueError(
#                 'The output of the "schedule" function ' "should be float."
#             )
#         if isinstance(lr, ops.Tensor) and not lr.dtype.is_floating:
#             raise ValueError("The dtype of Tensor should be float")
#         K.set_value(self.model.optimizer.lr, K.get_value(lr))
#         if self.verbose > 0:
#             print(
#                 "\nEpoch %05d: LearningRateScheduler reducing learning "
#                 "rate to %s." % (epoch + 1, lr)
#             )

#     def on_epoch_end(self, epoch, logs=None):
#         logs = logs or {}
#         logs["lr"] = K.get_value(self.model.optimizer.lr)


# @keras_export("keras.callbacks.ReduceLROnPlateau")
# class ReduceLROnPlateau(Callback):
#     """Reduce learning rate when a metric has stopped improving.

#   Models often benefit from reducing the learning rate by a factor
#   of 2-10 once learning stagnates. This callback monitors a
#   quantity and if no improvement is seen for a 'patience' number
#   of epochs, the learning rate is reduced.

#   Example:

#   ```python
#   reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
#                                 patience=5, min_lr=0.001)
#   model.fit(X_train, Y_train, callbacks=[reduce_lr])
#   ```

#   Arguments:
#       monitor: quantity to be monitored.
#       factor: factor by which the learning rate will be reduced. new_lr = lr *
#         factor
#       patience: number of epochs with no improvement after which learning rate
#         will be reduced.
#       verbose: int. 0: quiet, 1: update messages.
#       mode: one of {auto, min, max}. In `min` mode, lr will be reduced when the
#         quantity monitored has stopped decreasing; in `max` mode it will be
#         reduced when the quantity monitored has stopped increasing; in `auto`
#         mode, the direction is automatically inferred from the name of the
#         monitored quantity.
#       min_delta: threshold for measuring the new optimum, to only focus on
#         significant changes.
#       cooldown: number of epochs to wait before resuming normal operation after
#         lr has been reduced.
#       min_lr: lower bound on the learning rate.
#   """

#     def __init__(
#         self,
#         monitor="val_loss",
#         factor=0.1,
#         patience=10,
#         verbose=0,
#         mode="auto",
#         min_delta=1e-4,
#         cooldown=0,
#         min_lr=0,
#         **kwargs
#     ):
#         super(ReduceLROnPlateau, self).__init__()

#         self.monitor = monitor
#         if factor >= 1.0:
#             raise ValueError("ReduceLROnPlateau " "does not support a factor >= 1.0.")
#         if "epsilon" in kwargs:
#             min_delta = kwargs.pop("epsilon")
#             logging.warning(
#                 "`epsilon` argument is deprecated and "
#                 "will be removed, use `min_delta` instead."
#             )
#         self.factor = factor
#         self.min_lr = min_lr
#         self.min_delta = min_delta
#         self.patience = patience
#         self.verbose = verbose
#         self.cooldown = cooldown
#         self.cooldown_counter = 0  # Cooldown counter.
#         self.wait = 0
#         self.best = 0
#         self.mode = mode
#         self.monitor_op = None
#         self._reset()

#     def _reset(self):
#         """Resets wait counter and cooldown counter.
#     """
#         if self.mode not in ["auto", "min", "max"]:
#             logging.warning(
#                 "Learning Rate Plateau Reducing mode %s is unknown, "
#                 "fallback to auto mode.",
#                 self.mode,
#             )
#             self.mode = "auto"
#         if self.mode == "min" or (self.mode == "auto" and "acc" not in self.monitor):
#             self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
#             self.best = np.Inf
#         else:
#             self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
#             self.best = -np.Inf
#         self.cooldown_counter = 0
#         self.wait = 0

#     def on_train_begin(self, logs=None):
#         self._reset()

#     def on_epoch_end(self, epoch, logs=None):
#         logs = logs or {}
#         logs["lr"] = K.get_value(self.model.optimizer.lr)
#         current = logs.get(self.monitor)
#         if current is None:
#             logging.warning(
#                 "Reduce LR on plateau conditioned on metric `%s` "
#                 "which is not available. Available metrics are: %s",
#                 self.monitor,
#                 ",".join(list(logs.keys())),
#             )

#         else:
#             if self.in_cooldown():
#                 self.cooldown_counter -= 1
#                 self.wait = 0

#             if self.monitor_op(current, self.best):
#                 self.best = current
#                 self.wait = 0
#             elif not self.in_cooldown():
#                 self.wait += 1
#                 if self.wait >= self.patience:
#                     old_lr = float(K.get_value(self.model.optimizer.lr))
#                     if old_lr > self.min_lr:
#                         new_lr = old_lr * self.factor
#                         new_lr = max(new_lr, self.min_lr)
#                         K.set_value(self.model.optimizer.lr, new_lr)
#                         if self.verbose > 0:
#                             print(
#                                 "\nEpoch %05d: ReduceLROnPlateau reducing learning "
#                                 "rate to %s." % (epoch + 1, new_lr)
#                             )
#                         self.cooldown_counter = self.cooldown
#                         self.wait = 0

#     def in_cooldown(self):
#         return self.cooldown_counter > 0

