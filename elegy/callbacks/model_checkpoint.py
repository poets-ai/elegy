import os
import logging

import numpy as np
import six

from .callback import Callback


class ModelCheckpoint(Callback):
    """Callback to save the Elegy model or model weights at some frequency.

    `ModelCheckpoint` callback is used in conjunction with training using
    `model.fit()` to save a model or weights at some
    interval, so the model or weights can be loaded later to continue the training
    from the state saved.

    A few options this callback provides include:

    - Whether to only keep the model that has achieved the "best performance" so
        far, or whether to save the model at the end of every epoch regardless of
        performance.
    - Definition of 'best'; which quantity to monitor and whether it should be
        maximized or minimized.
    - The frequency it should save at. Currently, the callback supports saving at
        the end of every epoch, or after a fixed number of training batches.

    Example:

    ```python
    EPOCHS = 10
    checkpoint_filepath = '/tmp/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_acc',
        mode='max',
        save_best_only=True)

    # Model is saved at the end of every epoch, if it's the best seen
    # so far.
    model.fit(epochs=EPOCHS, callbacks=[model_checkpoint_callback])

    # The model status (that are considered the best) are loaded into the model.
    model.load(checkpoint_filepath)
    ```

    Arguments:
        filepath: string, path to save the model file. `filepath` can contain
            named formatting options, which will be filled the value of `epoch` and
            keys in `logs` (passed in `on_epoch_end`). For example: if `filepath` is
            `weights.{epoch:02d}-{val_loss:.2f}`, then the model checkpoints
            will be saved with the epoch number and the validation loss in the
            filename.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`, the latest best model according
            to the quantity monitored will not be overwritten.
            If `filepath` doesn't contain formatting options like `{epoch}` then
            `filepath` will be overwritten by each new better model.
        mode: one of {auto, min, max}. If `save_best_only=True`, the decision to
            overwrite the current save file is made based on either the maximization
            or the minimization of the monitored quantity. For `val_acc`, this
            should be `max`, for `val_loss` this should be `min`, etc. In `auto`
            mode, the direction is automatically inferred from the name of the
            monitored quantity.
        save_freq: `'epoch'` or integer. When using `'epoch'`, the callback saves
            the model after each epoch. When using integer, the callback saves the
            model at end of this many batches. Note that if the saving isn't aligned
            to epochs, the monitored metric may potentially be less reliable (it
            could reflect as little as 1 batch, since the metrics get reset every
            epoch). Defaults to `'epoch'`
    """

    def __init__(
        self,
        filepath,
        monitor="val_loss",
        verbose=0,
        save_best_only=False,
        mode="auto",
        save_freq="epoch",
        period=1,
    ):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.period = period
        self.epochs_since_last_save = 0
        self._batches_seen_since_last_saving = 0

        if mode not in ["auto", "min", "max"]:
            logging.warning(
                "ModelCheckpoint mode %s is unknown, " "fallback to auto mode.", mode
            )
            mode = "auto"

        if mode == "min":
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == "max":
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if "acc" in self.monitor or self.monitor.startswith("fmeasure"):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

        if self.save_freq != "epoch" and not isinstance(self.save_freq, int):
            raise ValueError("Unrecognized save_freq: {}".format(self.save_freq))

        # Only the chief worker writes model checkpoints, but all workers
        # restore checkpoint at on_train_begin().
        self._chief_worker_only = False

    def set_model(self, model):
        self.model = model

    def on_train_begin(self, logs=None):
        # pylint: disable=protected-access
        # if self.model._in_multi_worker_mode():
        #     # MultiWorkerTrainingState is used to manage the training state needed
        #     # for preemption-recovery of a worker in multi-worker training.
        #     self.model._training_state = training_state.MultiWorkerTrainingState(
        #         self.model, self.filepath
        #     )
        #     self._training_state = self.model._training_state
        #     if self._training_state.restore():
        #         # If the training state needs to be and is successfully restored,
        #         # it is recovering from a previous failure (or preemption). In such
        #         # case, do not load the weights from user specified file path.
        #         return

        # If this is not multi worker training, restoring is not needed, or
        # restoring failed, check if it should load weights on restart.
        # if self.load_weights_on_restart:
        # if (
        #     not self.model._in_multi_worker_mode()
        #     or multi_worker_util.should_load_checkpoint()
        # ):
        #     filepath_to_load = self._get_most_recently_modified_file_matching_pattern(
        #         self.filepath
        #     )
        #     if filepath_to_load is not None and training_state.checkpoint_exists(
        #         filepath_to_load
        #     ):
        #         try:
        #             # `filepath` may contain placeholders such as `{epoch:02d}`, and
        #             # thus it attempts to load the most recently modified file with file
        #             # name matching the pattern.
        #             self.model.load_weights(filepath_to_load)
        #         except (IOError, ValueError) as e:
        #             raise ValueError(
        #                 "Error loading file from {}. Reason: {}".format(
        #                     filepath_to_load, e
        #                 )
        #             )
        pass

    def on_train_end(self, logs=None):
        pass
        # pylint: disable=protected-access
        # if self.model._in_multi_worker_mode():
        #     if self.model.stop_training or getattr(
        #         self.model, "_successful_loop_finish", False
        #     ):
        #         # In multi-worker training, on successful exit of training, delete the
        #         # training state backup file that was saved for the purpose of worker
        #         # recovery.
        #         self._training_state.delete_backup()
        #         # Restore the training state so the model is ready for next (possible)
        #         # multi worker training.
        #         del self._training_state
        #         self.model._training_state = None

    def on_batch_end(self, batch, logs=None):
        if self._implements_train_batch_hooks():
            logs = logs or {}
            self._batches_seen_since_last_saving += 1
            if self._batches_seen_since_last_saving >= self.save_freq:
                self._save_model(epoch=self._current_epoch, logs=logs)
                self._batches_seen_since_last_saving = 0

    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        # pylint: disable=protected-access
        if self.save_freq == "epoch":
            # if self.model._in_multi_worker_mode():
            #     # Exclude training state variables in user-requested checkpoint file.
            #     with self._training_state.untrack_vars():
            #         self._save_model(epoch=epoch, logs=logs)
            # else:
            self._save_model(epoch=epoch, logs=logs)
        # if self.model._in_multi_worker_mode():
        #     # For multi-worker training, back up the weights and current training
        #     # state for possible future recovery.
        #     # TODO(rchao): Call `back_up` at finer period such as N steps.
        #     self._training_state.back_up(epoch)

    def _save_model(self, epoch, logs):
        """Saves the model.

    Arguments:
        epoch: the epoch this iteration is in.
        logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
    """
        logs = logs or {}

        if (
            isinstance(self.save_freq, int)
            or self.epochs_since_last_save >= self.period
        ):
            self.epochs_since_last_save = 0
            filepath = self._get_file_path(epoch, logs)

            # try:
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    logging.warning(
                        "Can save best model only with %s available, " "skipping.",
                        self.monitor,
                    )
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print(
                                "\nEpoch %05d: %s improved from %0.5f to %0.5f,"
                                " saving model to %s"
                                % (
                                    epoch + 1,
                                    self.monitor,
                                    self.best,
                                    current,
                                    filepath,
                                )
                            )
                        self.best = current
                        self.model.save(filepath)
                    else:
                        if self.verbose > 0:
                            print(
                                "\nEpoch %05d: %s did not improve from %0.5f"
                                % (epoch + 1, self.monitor, self.best)
                            )
            else:
                if self.verbose > 0:
                    print("\nEpoch %05d: saving model to %s" % (epoch + 1, filepath))

                self.model.save(filepath)

            self._maybe_remove_file()
            # except IOError as e:
            #     # `e.errno` appears to be `None` so checking the content of `e.args[0]`.
            #     if "is a directory" in six.ensure_str(e.args[0]):
            #         raise IOError(
            #             "Please specify a non-directory filepath for "
            #             "ModelCheckpoint. Filepath used is an existing "
            #             "directory: {}".format(filepath)
            #         )

    def _get_file_path(self, epoch, logs):
        """Returns the file path for checkpoint."""
        # pylint: disable=protected-access
        # if (
        #     not self.model._in_multi_worker_mode()
        #     or multi_worker_util.should_save_checkpoint()
        # ):
        try:
            # `filepath` may contain placeholders such as `{epoch:02d}` and
            # `{mape:.2f}`. A mismatch between logged metrics and the path's
            # placeholders can cause formatting to fail.
            return self.filepath.format(epoch=epoch + 1, **logs)
        except KeyError as e:
            raise KeyError(
                'Failed to format this callback filepath: "{}". '
                "Reason: {}".format(self.filepath, e)
            )
        # else:
        #     # If this is multi-worker training, and this worker should not
        #     # save checkpoint, we use a temp filepath to store a dummy checkpoint, so
        #     # it writes to a file that will be removed at the end of `_save_model()`
        #     # call. This is because the SyncOnReadVariable needs to be synced across
        #     # all the workers in order to be read, and all workers need to initiate
        #     # that.
        #     self._temp_file_dir = tempfile.mkdtemp()
        #     extension = os.path.splitext(self.filepath)[1]
        #     return os.path.join(self._temp_file_dir, "temp" + extension)

    def _maybe_remove_file(self):
        # Remove the checkpoint directory in multi-worker training where this worker
        # should not checkpoint. It is a dummy directory previously saved for sync
        # distributed training.
        pass
        # if (
        #     self.model._in_multi_worker_mode()
        #     and not multi_worker_util.should_save_checkpoint()  # pylint: disable=protected-access
        # ):
        #     file_io.delete_recursively(self._temp_file_dir)
        #     del self._temp_file_dir

    def _implements_train_batch_hooks(self):
        # If save_freq="epoch", batch-level hooks don't need to be run.
        return isinstance(self.save_freq, int)


# checkpoints_dir = os.path.join(save_dir, "checkpoints/epoch_{epoch:04d}/cp.ckpt")
