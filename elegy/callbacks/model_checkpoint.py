# Implementation based on tf.keras.callbacks.py
# https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/callbacks.py
import logging

import numpy as np

from .callback import Callback


class ModelCheckpoint(Callback):
    """
    Callback to save the Elegy model or model weights at some frequency.

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
    checkpoint_path = '/tmp/checkpoint'
    model_checkpoint_callback = elegy.callbacks.ModelCheckpoint(
        path=checkpoint_path,
        monitor='val_acc',
        mode='max',
        save_best_only=True)

    # Model is saved at the end of every epoch, if it's the best seen
    # so far.
    model.fit(epochs=EPOCHS, callbacks=[model_checkpoint_callback])

    # The model status (that are considered the best) are loaded into the model.
    model.load(checkpoint_path)
    ```

    """

    def __init__(
        self,
        path: str,
        monitor: str = "val_loss",
        verbose: int = 0,
        save_best_only: bool = False,
        mode: str = "auto",
        save_freq: str = "epoch",
        period: int = 1,
    ):
        """
        Arguments:
            path: string, path to directory to save the model state. `path` can contain
                named formatting options, which will be filled the value of `epoch` and
                keys in `logs` (passed in `on_epoch_end`). For example: if `path` is
                `weights.{epoch:02d}-{val_loss:.2f}`, then the model checkpoints
                will be saved with the epoch number and the validation loss in the
                filename.
            monitor: quantity to monitor.
            verbose: verbosity mode, 0 or 1.
            save_best_only: if `save_best_only=True`, the latest best model according
                to the quantity monitored will not be overwritten.
                If `path` doesn't contain formatting options like `{epoch}` then
                `path` will be overwritten by each new better model.
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
            period: the number of epochs between which the model is saved. This only works
                if `save_freq` is 'epoch', otherwise the `save_freq` will override
                this period.
        """
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.path = path
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

    def set_model(self, model):
        self.model = model

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
            self._save_model(epoch=epoch, logs=logs)

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
            path = self._get_file_path(epoch, logs)

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
                                    path,
                                )
                            )
                        self.best = current
                        self.model.save(path)
                    else:
                        if self.verbose > 0:
                            print(
                                "\nEpoch %05d: %s did not improve from %0.5f"
                                % (epoch + 1, self.monitor, self.best)
                            )
            else:
                if self.verbose > 0:
                    print("\nEpoch %05d: saving model to %s" % (epoch + 1, path))

                self.model.save(path)

            # except IOError as e:
            #     # `e.errno` appears to be `None` so checking the content of `e.args[0]`.
            #     if "is a directory" in six.ensure_str(e.args[0]):
            #         raise IOError(
            #             "Please specify a non-directory path for "
            #             "ModelCheckpoint. Filepath used is an existing "
            #             "directory: {}".format(path)
            #         )

    def _get_file_path(self, epoch, logs):
        """Returns the file path for checkpoint."""
        # pylint: disable=protected-access
        try:
            # `path` may contain placeholders such as `{epoch:02d}` and
            # `{mape:.2f}`. A mismatch between logged metrics and the path's
            # placeholders can cause formatting to fail.
            return self.path.format(epoch=epoch + 1, **logs)
        except KeyError as e:
            raise KeyError(
                'Failed to format this callback path: "{}". '
                "Reason: {}".format(self.path, e)
            )

    def _implements_train_batch_hooks(self):
        # If save_freq="epoch", batch-level hooks don't need to be run.
        return isinstance(self.save_freq, int)
