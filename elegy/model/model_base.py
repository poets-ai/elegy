from elegy.utils import Mode
import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
import optax

from elegy import module, utils
from elegy.losses.loss import Loss
from elegy.metrics.metric import Metric
from elegy.module import Module
from elegy.module import jit as elegy_jit


class ModelBase(Module):
    def __init__(
        self,
        module: tp.Optional[Module] = None,
        loss: tp.Union[tp.Callable, tp.List, tp.Dict, None] = None,
        metrics: tp.Union[tp.Callable, tp.List, tp.Dict, None] = None,
        optimizer: tp.Optional[optax.GradientTransformation] = None,
        run_eagerly: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.module = module
        self.loss = Losses(loss) if loss is not None else None
        self.metrics = Metrics(metrics) if metrics else None
        self.optimizer = Optimizer(optimizer) if optimizer is not None else None
        self._predict_step_jit = elegy_jit(self.predict_fn, modules=self)
        self._test_step_jit = elegy_jit(self.test_fn, modules=self)
        self._train_step_jit = elegy_jit(self.train_fn, modules=self)
        self.initial_metrics_state: tp.Optional[tp.Dict[str, tp.Any]] = None
        self.run_eagerly = run_eagerly

        if self.module is not None:
            utils.wraps(self.module)(self)
        else:
            utils.wraps(self.call)(self)

    def call(self, *args, **kwargs):
        if self.module is not None:
            return self.module(*args, **kwargs)
        else:
            raise NotImplementedError("Must provide 'module' or implement 'call'.")

    def reset_metrics(self, hard: bool = False):
        if hard:
            self.metrics.reset()
            self.initial_metrics_state = None
        elif self.initial_metrics_state is not None:
            self.metrics.set_parameters(self.initial_metrics_state)

    def predict_fn(self, x: tp.Any = ()):

        x_args, x_kwargs = utils.get_input_args(x, training=module.is_training())
        y_pred = utils.inject_dependencies(self)(*x_args, **x_kwargs)

        return y_pred

    def predict_step(self, x: tp.Any = ()):
        with module.context(training=False, hooks=False):
            return self.predict_fn(x=x)

    def predict_step_jit(self, x: tp.Any = ()):
        with module.context(training=False, hooks=False):
            return self._predict_step_jit(x)

    def predict_on_batch(
        self, x: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple]
    ) -> tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple]:
        """
        Returns predictions for a single batch of samples.

        Arguments:
            x: Input data. A Numpy/Jax array (or array-like), or possibly
                nested python structure of dict, list, tuple that contain
                arrays as leafs.

        Returns:
            Jax array(s) of predictions.

        Raises:
            ValueError: In case of mismatch between given number of inputs and
                expectations of the model.
        """
        self.maybe_initialize(mode=Mode.predict, x=x)

        method = self.predict_step if self.run_eagerly else self.predict_step_jit

        return method(x=x)

    def loss_fn(
        self,
        x: tp.Any = (),
        y: tp.Any = None,
        sample_weight: tp.Optional[np.ndarray] = None,
        class_weight: tp.Optional[np.ndarray] = None,
    ):
        y_pred = self.predict_fn(x)

        if self.loss is not None:
            loss_logs = self.loss(
                x=x,
                y_true=y,
                y_pred=y_pred,
                sample_weight=sample_weight,
                class_weight=class_weight,
                training=module.is_training(),
                parameters=self.module.get_parameters(trainable=True),
                states=self.module.get_parameters(trainable=False),
            )
        else:
            loss_logs = {}

        hooks_losses_logs = module.get_losses()

        if hooks_losses_logs is None:
            hooks_losses_logs = {}

        loss = sum(loss_logs.values()) + sum(hooks_losses_logs.values())

        total_loss_logs = {}
        total_loss_logs.update(hooks_losses_logs)
        total_loss_logs.update(loss_logs)
        total_loss_logs["loss"] = loss

        return loss, y_pred, total_loss_logs

    def test_fn(
        self,
        x: tp.Any = (),
        y: tp.Any = None,
        sample_weight: tp.Optional[np.ndarray] = None,
        class_weight: tp.Optional[np.ndarray] = None,
        get_gradients: bool = False,
    ) -> tp.Tuple[np.ndarray, tp.Dict, tp.Optional[tp.Dict]]:

        if get_gradients:
            (loss, y_pred, total_loss_logs), grads = module.value_and_grad(
                self.loss_fn, modules=self.module
            )(x, y, sample_weight, class_weight)
        else:
            grads = None
            loss, y_pred, total_loss_logs = self.loss_fn(
                x, y, sample_weight, class_weight
            )

        if self.metrics is not None:
            logs = self.metrics(
                total_loss_logs,
                x=x,
                y_true=y,
                y_pred=y_pred,
                sample_weight=sample_weight,
                class_weight=class_weight,
                training=module.is_training(),
                parameters=self.module.get_parameters(trainable=True),
                states=self.module.get_parameters(trainable=False),
            )
        else:
            logs = {}

        return loss, logs, grads

    def test_step(
        self,
        x: tp.Any = (),
        y: tp.Any = None,
        sample_weight: tp.Optional[np.ndarray] = None,
        class_weight: tp.Optional[np.ndarray] = None,
        get_gradients: bool = False,
    ) -> tp.Tuple[np.ndarray, tp.Dict, tp.Optional[tp.Dict]]:

        with module.context(training=False, hooks=True):
            return self.test_fn(
                x=x,
                y=y,
                sample_weight=sample_weight,
                class_weight=class_weight,
                get_gradients=get_gradients,
            )

    def test_step_jit(
        self,
        x: tp.Any = (),
        y: tp.Any = None,
        sample_weight: tp.Optional[np.ndarray] = None,
        class_weight: tp.Optional[np.ndarray] = None,
    ):
        with module.context(training=False, hooks=True):
            return self._test_step_jit(x, y, sample_weight, class_weight)

    def test_on_batch(
        self,
        x: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple],
        y: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple, None] = None,
        sample_weight: tp.Optional[jnp.ndarray] = None,
        class_weight: tp.Optional[jnp.ndarray] = None,
    ) -> tp.Dict[str, jnp.ndarray]:
        """
        Test the model on a single batch of samples.

        Arguments:
            x: Input data. It could be:

                - A Numpy array (or array-like), or a list
                    of arrays (in case the model has multiple inputs).
                - A dict mapping input names to the corresponding arrays, if
                    the model has named inputs.
            y: Target data. Like the input data `x`, it could be either Numpy
                array(s) or Jax array(s).
            sample_weight: Optional array of the same length as x, containing
                weights to apply to the model's loss for each sample. In the case of
                temporal data, you can pass a 2D array with shape (samples,
                sequence_length), to apply a different weight to every timestep of
                every sample.

        Returns:
            A `logs` dictionary of containing the main `loss` as well as all
            other losses and metrics.
        Raises:
            ValueError: In case of invalid user-provided arguments.
        """
        self.maybe_initialize(
            mode=Mode.test,
            x=x,
            y=y,
            sample_weight=sample_weight,
            class_weight=class_weight,
        )

        method = self.test_step if self.run_eagerly else self.test_step_jit

        loss, logs, grads = method(
            x=x, y=y, sample_weight=sample_weight, class_weight=class_weight
        )

        return logs

    def train_fn(
        self,
        x: tp.Any = (),
        y: tp.Any = None,
        sample_weight: tp.Optional[np.ndarray] = None,
        class_weight: tp.Optional[np.ndarray] = None,
    ) -> tp.Dict[str, tp.Any]:
        assert self.optimizer is not None

        print("train_fn")

        loss, logs, grads = self.test_fn(
            x=x,
            y=y,
            sample_weight=sample_weight,
            class_weight=class_weight,
            get_gradients=True,
        )

        assert grads is not None

        parameters = self.module.get_parameters(trainable=True)

        parameters = self.optimizer(parameters, grads)

        if not module.is_initializing():
            self.module.set_parameters(parameters)

        return logs

    def train_step(
        self,
        x: tp.Any = (),
        y: tp.Any = None,
        sample_weight: tp.Optional[np.ndarray] = None,
        class_weight: tp.Optional[np.ndarray] = None,
    ) -> tp.Dict[str, tp.Any]:

        with module.context(training=True, hooks=True):
            return self.train_fn(
                x=x, y=y, sample_weight=sample_weight, class_weight=class_weight
            )

    def train_step_jit(
        self,
        x: tp.Any = (),
        y: tp.Any = None,
        sample_weight: tp.Optional[np.ndarray] = None,
        class_weight: tp.Optional[np.ndarray] = None,
    ):
        with module.context(training=True, hooks=True):
            outputs = self._train_step_jit(x, y, sample_weight, class_weight)

        return outputs

    def train_on_batch(
        self,
        x: tp.Union[np.ndarray, tp.Mapping[str, tp.Any], tp.Tuple],
        y: tp.Union[np.ndarray, tp.Mapping[str, tp.Any], tp.Tuple, None] = None,
        sample_weight: tp.Optional[np.ndarray] = None,
        class_weight: tp.Optional[np.ndarray] = None,
    ) -> tp.Dict[str, np.ndarray]:
        """
        Runs a single gradient update on a single batch of data.

        Arguments:
            x: Input data. It could be:

                - A Numpy array (or array-like), or a iterable of arrays
                    (in case the model has multiple inputs).
                - A dict mapping input names to the corresponding arrays,
                    if the model has named inputs.
            y: Target data. Like the input data `x`, it could be either Numpy
                array(s) or Jax array(s). It should be consistent with `x`
                (you cannot have Numpy inputs and array targets, or inversely).
            sample_weight: Optional array of the same length as x, containing
                weights to apply to the model's loss for each sample. In the case of
                temporal data, you can pass a 2D array with shape (samples,
                sequence_length), to apply a different weight to every timestep of
                every sample.
            class_weight: Optional dictionary mapping class indices (integers) to a
                weight (float) to apply to the model's loss for the samples from this
                class during training. This can be useful to tell the model to "pay
                more attention" to samples from an under-represented class.

        Returns:
            A `logs` dictionary of containing the main `loss` as well as all
            other losses and metrics.

        Raises:
            ValueError: In case of invalid user-provided arguments.
        """
        self.maybe_initialize(
            mode=Mode.train,
            x=x,
            y=y,
            sample_weight=sample_weight,
            class_weight=class_weight,
        )

        method = self.train_step if self.run_eagerly else self.train_step_jit

        return method(x=x, y=y, sample_weight=sample_weight, class_weight=class_weight)

    def maybe_initialize(
        self,
        mode: Mode,
        x: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple] = (),
        y: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple, None] = None,
        sample_weight: tp.Optional[jnp.ndarray] = None,
        class_weight: tp.Optional[jnp.ndarray] = None,
    ):

        with module.init_context(), module.context(training=True, hooks=True):

            if not self.module.initialized:
                self.predict_fn(x=x)
                self.module.initialized = True

            if mode == Mode.test and not self.metrics.initialized:

                self.test_fn(
                    x=x,
                    y=y,
                    sample_weight=sample_weight,
                    class_weight=class_weight,
                )
                self.metrics.initialized = True

                self.initial_metrics_state = self.metrics.get_parameters(
                    trainable=False
                )

            elif mode == Mode.train and not self.optimizer.initialized:
                self.train_fn(
                    x=x,
                    y=y,
                    sample_weight=sample_weight,
                    class_weight=class_weight,
                )
                self.metrics.initialized = True
                self.optimizer.initialized = True

                if self.initial_metrics_state is None:
                    self.initial_metrics_state = self.metrics.get_parameters(
                        trainable=False
                    )


class Optimizer(Module):
    def __init__(self, optimizer: optax.GradientTransformation, **kwargs):
        super().__init__(**kwargs)
        self.optax_optimizer = optimizer

    def call(self, parameters, grads):

        optimizer_state = self.add_parameter(
            "optimizer_state",
            initializer=lambda *args: self.optax_optimizer.init(parameters),
            trainable=False,
        )

        updates, optimizer_state = self.optax_optimizer.update(
            grads, optimizer_state, parameters
        )

        parameters = optax.apply_updates(parameters, updates)

        self.update_parameter("optimizer_state", optimizer_state)

        return parameters


class Losses(Module):
    def __init__(self, losses):
        super().__init__(name="losses")
        self.losses = losses

    def call(self, **kwargs):

        logs = {}

        for context, val in self.apply_recursive((), self.losses, **kwargs):
            loss_name = self.get_unique_loss_name(context, logs)
            logs[loss_name] = val

        return logs

    def apply_recursive(self, context: tp.Tuple[str, ...], losses, **kwargs):

        if isinstance(losses, tp.Callable):
            name = (
                losses.name
                if isinstance(losses, Loss)
                else utils.lower_snake_case(losses.__name__)
            )
            context += (name,)
            val = utils.inject_dependencies(losses)(**kwargs)

            if isinstance(val, tp.Dict):
                for name, val in val.items():
                    yield context + (name,), val
            else:
                yield context, val

        elif isinstance(losses, (tp.Tuple, tp.List)):
            for loss in losses:
                yield from self.apply_recursive(context, loss, **kwargs)
        elif isinstance(losses, tp.Dict):
            for name, loss in losses.items():
                yield from self.apply_recursive(context + (name,), loss, **kwargs)
        else:
            raise TypeError(f"Invalid type {type(losses)}")

    def get_unique_loss_name(self, context, logs):
        context = list(context)

        if not context[0].endswith("loss"):
            context[0] += "_loss"

        name = "/".join(context)

        if name not in logs:
            return name

        i = 1
        while f"{name}_{i}" in logs:
            i += 1

        return f"{name}_{i}"


class LossMetrics(Metric):
    def call(self, logs):

        count = self.add_parameter("count", initializer=jnp.zeros, trainable=False)
        total = self.add_parameter(
            "total",
            initializer=lambda *args: jax.tree_map(lambda x: jnp.array(0.0), logs),
            trainable=False,
        )

        count += 1
        total = jax.tree_multimap(lambda a, b: a + b, total, logs)

        self.update_parameter("count", count)
        self.update_parameter("total", total)

        logs = jax.tree_map(lambda total: total / count, total)

        return logs


class Metrics(Metric):
    def __init__(self, metrics, **kwargs):
        super().__init__(**kwargs)
        self.metrics = metrics

    def call(self, logs, **kwargs):

        # Loss logs
        logs = LossMetrics()(logs)

        # Metric logs
        for context, val in self.apply_recursive((), self.metrics, **kwargs):
            name = "/".join(context)
            name = self.get_unique_metric_name(logs, name)
            logs[name] = val

        return logs

    def apply_recursive(self, context: tp.Tuple[str, ...], metrics, **kwargs):

        if isinstance(metrics, tp.Callable):

            name = (
                metrics.name
                if isinstance(metrics, module.Module)
                else utils.lower_snake_case(metrics.__name__)
            )
            context += (name,)
            value = utils.inject_dependencies(metrics)(**kwargs)

            if isinstance(value, tp.Dict):
                for name, value in value.items():
                    yield context + (name,), value
            else:
                yield context, value

        elif isinstance(metrics, (tp.Tuple, tp.List)):
            for loss in metrics:
                yield from self.apply_recursive(context, loss, **kwargs)
        elif isinstance(metrics, tp.Dict):
            for name, loss in metrics.items():
                yield from self.apply_recursive(context + (name,), loss, **kwargs)
        else:
            raise TypeError(f"Invalid type {type(metrics)}")

    def get_unique_metric_name(self, logs, name):

        if name not in logs:
            return name

        i = 1
        while f"{name}_{i}" in logs:
            i += 1

        return f"{name}_{i}"
