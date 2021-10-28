# Implementation based on tf.keras.callbacks.py
# https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/callbacks.py
import enum
import logging
import signal
import typing as tp

import numpy as np

from .callback import Callback

ORIGINAL_HANDLER = signal.getsignal(signal.SIGINT)


class SigIntMode(enum.Enum):
    TRAIN = enum.auto()
    TEST = enum.auto()


class SigInt(Callback):
    def __init__(self, mode: tp.Union[SigIntMode, str]):
        super().__init__()
        self.mode = mode if isinstance(mode, SigIntMode) else SigIntMode(mode.upper())

    def signal_handler(self, signal, frame):
        print("\n\nStopping...\n\n")
        self.model.stop_training = True
        # signal.signal(signal.SIGINT, ORIGINAL_HANDLER)

    def on_train_begin(self, logs=None):
        signal.signal(signal.SIGINT, self.signal_handler)

    def on_train_end(self, logs=None):
        if self.mode == SigIntMode.TRAIN:
            signal.signal(signal.SIGINT, ORIGINAL_HANDLER)

    def on_test_begin(self, logs=None):
        signal.signal(signal.SIGINT, self.signal_handler)

    def on_test_end(self, logs=None):
        if self.mode == SigIntMode.TEST:
            signal.signal(signal.SIGINT, ORIGINAL_HANDLER)
