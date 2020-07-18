# Implementation based on tf.keras.callbacks.py
# https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/callbacks.py
try:
    import requests
except ImportError:
    requests = None
import json
import logging
import typing as tp

import numpy as np

from .callback import Callback


class RemoteMonitor(Callback):
    """Callback used to stream events to a server.

    Requires the `requests` library.
    Events are sent to `root + '/publish/epoch/end/'` by default. Calls are
    HTTP POST, with a `data` argument which is a
    JSON-encoded dictionary of event data.
    If send_as_json is set to True, the content type of the request will be
    application/json. Otherwise the serialized JSON will be sent within a form.
    """

    def __init__(
        self,
        root: str = "http://localhost:9000",
        path: str = "/publish/epoch/end/",
        field: str = "data",
        headers: tp.Optional[tp.Dict[str, str]] = None,
        send_as_json: bool = False,
    ):
        """
        Arguments:
            root: String; root url of the target server.
            path: String; path relative to `root` to which the events will be sent.
            field: String; JSON field under which the data will be stored.
                The field is used only if the payload is sent within a form
                (i.e. send_as_json is set to False).
            headers: Dictionary; optional custom HTTP headers.
            send_as_json: Boolean; whether the request should be
                sent as application/json.
        """
        super(RemoteMonitor, self).__init__()

        self.root = root
        self.path = path
        self.field = field
        self.headers = headers
        self.send_as_json = send_as_json

    def on_epoch_end(self, epoch, logs=None):
        if requests is None:
            raise ImportError("RemoteMonitor requires the `requests` library.")
        logs = logs or {}
        send = {}
        send["epoch"] = epoch
        for k, v in logs.items():
            # np.ndarray and np.generic are not scalar types
            # therefore we must unwrap their scalar values and
            # pass to the json-serializable dict 'send'
            if isinstance(v, (np.ndarray, np.generic)):
                send[k] = v.item()
            else:
                send[k] = v
        try:
            if self.send_as_json:
                requests.post(self.root + self.path, json=send, headers=self.headers)
            else:
                requests.post(
                    self.root + self.path,
                    {self.field: json.dumps(send)},
                    headers=self.headers,
                )
        except requests.exceptions.RequestException:
            logging.warning(
                "Warning: could not reach RemoteMonitor "
                "root server at " + str(self.root)
            )
