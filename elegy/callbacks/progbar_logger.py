# Implementation based on tf.keras.callbacks.py and tf.keras.utils.generic_utils.py
# https://github.com/tensorflow/tensorflow/blob/2b96f3662bd776e277f86997659e61046b56c315/tensorflow/python/keras/callbacks.py
# https://github.com/tensorflow/tensorflow/blob/2b96f3662bd776e277f86997659e61046b56c315/tensorflow/python/keras/utils/generic_utils.py

import copy
import os
import sys
import time

import numpy as np

from .callback import Callback


class ProgbarLogger(Callback):
    """Callback that prints metrics to stdout.

    Arguments:
        count_mode: One of "steps" or "samples".
            Whether the progress bar should
            count samples seen or steps (batches) seen.
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over an epoch.
            Metrics in this list will be logged as-is.
            All others will be averaged over time (e.g. loss, etc).
            If not provided, defaults to the `Model`'s metrics.

    Raises:
        ValueError: In case of invalid `count_mode`.
    """

    def __init__(self, count_mode="samples", stateful_metrics=None):
        super(ProgbarLogger, self).__init__()
        if count_mode == "samples":
            self.use_steps = False
        elif count_mode == "steps":
            self.use_steps = True
        else:
            raise ValueError("Unknown `count_mode`: " + str(count_mode))
        # Defaults to all Model's metrics except for loss.
        self.stateful_metrics = set(stateful_metrics) if stateful_metrics else None

        self.seen = 0
        self.progbar = None
        self.target = None
        self.verbose = 1
        self.epochs = 1

        self._called_in_fit = False

    def set_params(self, params):
        self.verbose = params["verbose"]
        self.epochs = params["epochs"]
        if self.use_steps and "steps" in params:
            self.target = params["steps"]
        elif not self.use_steps and "samples" in params:
            self.target = params["samples"]
        else:
            self.target = None  # Will be inferred at the end of the first epoch.

    def on_train_begin(self, logs=None):
        # When this logger is called inside `fit`, validation is silent.
        self._called_in_fit = True

    def on_test_begin(self, logs=None):
        if not self._called_in_fit:
            self._reset_progbar()

    def on_predict_begin(self, logs=None):
        self._reset_progbar()

    def on_epoch_begin(self, epoch, logs=None):
        self._reset_progbar(epoch=epoch)
        if self.verbose in [1, 2] and self.epochs > 1:
            print("Epoch %d/%d" % (epoch + 1, self.epochs))

    def on_train_batch_end(self, batch, logs=None):
        self._batch_update_progbar(logs)

    def on_test_batch_end(self, batch, logs=None):
        if not self._called_in_fit:
            self._batch_update_progbar(logs)

    def on_predict_batch_end(self, batch, logs=None):
        self._batch_update_progbar(None)  # Don't pass prediction results.

    def on_epoch_end(self, epoch, logs=None):
        self._finalize_progbar(logs)

    def on_test_end(self, logs=None):
        if not self._called_in_fit:
            self._finalize_progbar(logs)

    def on_predict_end(self, logs=None):
        self._finalize_progbar(logs)

    def _reset_progbar(self, epoch=None):
        self.seen = 0

        prevprogbar = self.progbar
        self.progbar = None
        self.progbar = Progbar(
            target=self.target,
            verbose=self.verbose,
            stateful_metrics=self.stateful_metrics,
            unit_name="step" if self.use_steps else "sample",
            epoch=epoch,
        )
        if prevprogbar is not None:
            # inherit the column widths from the previous progbar for nicer looks
            self.progbar._compact_table_column_width = (
                prevprogbar._compact_table_column_width
            )

    def _batch_update_progbar(self, logs=None):
        """Updates the progbar."""
        if self.stateful_metrics is None:
            if logs:
                # self.stateful_metrics = set(m.name for m in self.model.metrics)
                self.stateful_metrics = set(logs.keys())
            else:
                self.stateful_metrics = set()

        logs = copy.copy(logs) if logs else {}
        batch_size = logs.pop("size", 0)
        num_steps = logs.pop("num_steps", 1)  # DistStrat can run >1 steps.
        logs.pop("batch", None)
        add_seen = num_steps if self.use_steps else num_steps * batch_size
        self.seen += add_seen
        self.progbar.update(self.seen, list(logs.items()), finalize=False)

    def _finalize_progbar(self, logs):
        if self.target is None:
            self.target = self.seen
            self.progbar.target = self.seen
        logs = logs or {}
        # remove size or val_size for displaying
        logs.pop("size", None)
        logs.pop("val_size", None)
        self.progbar.update(self.seen, list(logs.items()), finalize=True)


class Progbar(object):
    """Displays a progress bar.

    Arguments:
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose), 3 (compact table)
        stateful_metrics: Iterable of string names of metrics that should *not* be
          averaged over time. Metrics in this list will be displayed as-is. All
          others will be averaged by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
        unit_name: Display name for step counts (usually "step" or "sample").
    """

    def __init__(
        self,
        target,
        width=30,
        verbose=1,
        interval=0.05,
        stateful_metrics=None,
        unit_name="step",
        epoch=None,
    ):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        self.unit_name = unit_name
        self.epoch = epoch
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = (
            (hasattr(sys.stdout, "isatty") and sys.stdout.isatty())
            or "ipykernel" in sys.modules
            or "posix" in sys.modules
            or "PYCHARM_HOSTED" in os.environ
        )
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0
        self._compact_header_already_printed_names = []
        self._compact_table_column_width = dict()

    def update(self, current, values=None, finalize=None):
        """Updates the progress bar.

        Arguments:
            current: Index of current step.
            values: List of tuples: `(name, value_for_last_step)`. If `name` is in
              `stateful_metrics`, `value_for_last_step` will be displayed as-is.
              Else, an average of the metric over time will be displayed.
            finalize: Whether this is the last update for the progress bar. If
              `None`, defaults to `current >= self.target`.
        """
        if finalize is None:
            if self.target is None:
                finalize = False
            else:
                finalize = current >= self.target

        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                # In the case that progress bar doesn't have a target value in the first
                # epoch, both on_batch_end and on_epoch_end will be called, which will
                # cause 'current' and 'self._seen_so_far' to have the same value. Force
                # the minimal value to 1 here, otherwise stateful_metric will be 0s.
                value_base = max(current - self._seen_so_far, 1)
                if k not in self._values:
                    self._values[k] = [v * value_base, value_base]
                else:
                    self._values[k][0] += v * value_base
                    self._values[k][1] += value_base
            else:
                # Stateful metrics output a numeric value. This representation
                # means "take an average from a single value" but keeps the
                # numeric formatting.
                self._values[k] = [v, 1]
        self._seen_so_far = current

        now = time.time()
        info = " - %.0fs" % (now - self._start)
        if self.verbose == 1:
            if now - self._last_update < self.interval and not finalize:
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write("\b" * prev_total_width)
                sys.stdout.write("\r")
            else:
                sys.stdout.write("\n")

            if self.target is not None:
                numdigits = int(np.log10(self.target)) + 1
                bar = ("%" + str(numdigits) + "d/%d [") % (current, self.target)
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += "=" * (prog_width - 1)
                    if current < self.target:
                        bar += ">"
                    else:
                        bar += "="
                bar += "." * (self.width - prog_width)
                bar += "]"
            else:
                bar = "%7d/Unknown" % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0

            if self.target is None or finalize:
                if time_per_unit >= 1 or time_per_unit == 0:
                    info += " %.0fs/%s" % (time_per_unit, self.unit_name)
                elif time_per_unit >= 1e-3:
                    info += " %.0fms/%s" % (time_per_unit * 1e3, self.unit_name)
                else:
                    info += " %.0fus/%s" % (time_per_unit * 1e6, self.unit_name)
            else:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = "%d:%02d:%02d" % (
                        eta // 3600,
                        (eta % 3600) // 60,
                        eta % 60,
                    )
                elif eta > 60:
                    eta_format = "%d:%02d" % (eta // 60, eta % 60)
                else:
                    eta_format = "%ds" % eta

                info = " - ETA: %s" % eta_format

            for k in self._values_order:
                info += " - %s:" % k
                if isinstance(self._values[k], list):
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += " %.4f" % avg
                    else:
                        info += " %.4e" % avg
                else:
                    info += " %s" % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += " " * (prev_total_width - self._total_width)

            if finalize:
                info += "\n"

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if finalize:
                numdigits = int(np.log10(self.target)) + 1
                count = ("%" + str(numdigits) + "d/%d") % (current, self.target)
                info = count + info
                for k in self._values_order:
                    info += " - %s:" % k
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += " %.4f" % avg
                    else:
                        info += " %.4e" % avg
                info += "\n"

                sys.stdout.write(info)
                sys.stdout.flush()

        elif self.verbose == 3:
            self.compact_table_progress(current, finalize)

        elif self.verbose == 4 and finalize:
            self.compact_table_progress(current, finalize)

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)

    def compact_table_progress(self, current, finalize=False):
        now = time.time()
        if now - self._last_update < self.interval and not finalize:
            return

        def to_column_name(i):
            # 0,1,2,3,4,5,6,7,8,9,A,B,C,D, etc
            return str(i) if i < 10 else chr(i + 55)

        if self.epoch == 0 and (self._last_update == 0 or finalize):
            # first epoch, first or last update: show the metric names
            info = ""
            for i, k in enumerate(["Step/Epoch", "Time"] + self._values_order):
                if k not in self._compact_header_already_printed_names:
                    i = to_column_name(i)
                    info += f"[{i}]{k} | "
                    self._compact_header_already_printed_names += [k]
            # remove the last |
            info = info[:-3]
            if len(info):
                print(info, end="\n", file=sys.stdout, flush=True)

        info = " "
        # first column: show step if running, epoch if finished
        if finalize and self.epoch is not None:
            # epoch
            numdigits = int(np.log10(self.target)) + 1
            colstr = f"%{numdigits*2+1}d" % self.epoch
        else:
            # step
            if self.target is not None:
                numdigits = int(np.log10(self.target)) + 1
                colstr = ("%" + str(numdigits) + "d/%d") % (current, self.target)
                progress = float(current) / self.target
            else:
                colstr = "%7d/Unknown" % current
        self._compact_table_column_width[0] = len(colstr)
        info += colstr

        # second column: elapsed time
        elapsed = now - self._start
        timestr = f"{elapsed:6.1f}s"
        self._compact_table_column_width[1] = len(timestr)
        info += " | " + timestr

        for i, k in enumerate(self._values_order):
            avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
            if 1e4 >= avg > 1e-3:
                valstr = "%.3f" % avg
            else:
                valstr = "%.3e" % avg

            # prepend spaces if necessary to keep column width the same as before
            valstr = (
                " " * (self._compact_table_column_width.get(i + 2, 0) - len(valstr))
                + valstr
            )
            self._compact_table_column_width[i + 2] = len(valstr)
            info += " | " + valstr

        if self.epoch == 0 and finalize:
            # first epoch, last update: show table separator with column numbers
            if self.target is None:
                self.target = current

            sep = " " * len(info) + "\n"
            for i in range(len(self._values_order) + 2):
                colwidth = self._compact_table_column_width[i]
                colname = to_column_name(i)
                colwidth = colwidth - len(f"[{colname}]")
                sep += (
                    "#" * (int(np.ceil(colwidth / 2)) + 1)
                    + f"[{colname}]"
                    + "#" * (int(np.floor(colwidth / 2)) + 1)
                    + "|"
                )

            # remove the last |
            sep = sep[:-1]
            print(sep, end="\n", file=sys.stdout, flush=True)

        print(
            info,
            end="\r" if self._dynamic_display and not finalize else "\n",
            file=sys.stdout,
            flush=True,
        )
