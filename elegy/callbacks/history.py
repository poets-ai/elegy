# Implementation based on tf.keras.callbacks.py
# https://github.com/tensorflow/tensorflow/blob/2b96f3662bd776e277f86997659e61046b56c315/tensorflow/python/keras/callbacks.py

from .callback import Callback


class History(Callback):
    """Callback that records events into a `History` object.

  This callback is automatically applied to
  every Keras model. The `History` object
  gets returned by the `fit` method of models.
  """

    def __init__(self):
        super(History, self).__init__()
        self.history = {}

    def on_train_begin(self, logs=None):
        self.epoch = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        # Set the history attribute on the model after the epoch ends. This will
        # make sure that the state which is set is the latest one.
        self.model.history = self
