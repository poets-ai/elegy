# Implementation based on tf.keras.callbacks.py
# https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/callbacks.py
import numpy as np

from .callback import Callback


class TerminateOnNaN(Callback):
    """Callback that terminates training when a NaN loss is encountered."""

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                print("Batch %d: Invalid loss, terminating training" % (batch))
                self.model.stop_training = True
