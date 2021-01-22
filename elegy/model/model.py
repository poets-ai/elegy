from elegy.optimizer import Optimizer
import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
from elegy import hooks, utils, module
from elegy.model.generalized_module.generalized_module import (
    GeneralizedModule,
    generalize,
)
from elegy.model.generalized_optimizer.generalized_optimizer import (
    GeneralizedOptimizer,
    generalize_optimizer,
)
from elegy.model.model_base import ModelBase
from elegy.model.model_core import Prediction, States
from elegy.types import (
    Backprop,
    MissingOptimizer,
    RNG,
    Evaluation,
    Logs,
    MissingModule,
    OutputStates,
    Prediction,
    Scalar,
    Training,
    UNINITIALIZED,
)
from elegy.types import Mode, RNGSeq, Uninitialized
from jax._src.random import t


class Model(ModelBase):
    module: tp.Any = None
    loss: tp.Any = None
    metrics: tp.Any = None
    optimizer: tp.Any = None
    seed: int = 42

    module_internal: tp.Optional[GeneralizedModule]
    loss_internal: "Losses"
    metrics_internal: "Metrics"
    optimizer_internal: tp.Optional[GeneralizedOptimizer]

    def __init__(
        self,
        module: tp.Any = None,
        loss: tp.Any = None,
        metrics: tp.Any = None,
        optimizer: tp.Any = None,
        seed: int = 42,
        **kwargs,
    ):
        """[summary]

        Arguments:
            module: A `Module` instance.
            loss: A `elegy.Loss` or `Callable` instance representing the loss function of the network.
                You can define more loss terms by simply passing a possibly nested structure of
                lists and dictionaries of `elegy.Loss` or `Callable`s. Usually a plain list of losses is enough
                but using dictionaries will create namescopes for the names of the losses
                which might be useful e.g. to group things in tensorboard. Contrary to Keras convention,
                in Elegy there is no relation between the structure of `loss` with the structure
                of the labels and outputs of the network. Elegy's loss system is more flexible than
                the one provided by Keras, for more information on how to mimick Keras behavior checkout the
                [Losses and Metrics Guide](https://poets-ai.github.io/elegy/guides/losses-and-metrics)`.
            metrics: A `elegy.Metric` or `Callable` instance representing the loss function of the network.
                You can define more metrics terms by simply passing a possibly nested structure of
                lists and dictionaries of `elegy.Metric` or `Callable`s. Usually a plain list of metrics is enough
                but using dictionaries will create namescopes for the names of the metrics
                which might be useful e.g. to group things in tensorboard. Contrary to Keras convention,
                in Elegy there is no relation between the structure of `metrics` with the structure
                of the labels and outputs of the network. Elegy's metrics system is more flexible than
                the one provided by Keras, for more information on how to mimick Keras behavior checkout the
                [Losses and Metrics Guide](https://poets-ai.github.io/elegy/guides/losses-and-metrics)`.
            optimizer: A `optax` optimizer instance. Optix is a very flexible library for defining
                optimization pipelines with things like learning rate schedules, this means that
                there is no need for a `LearningRateScheduler` callback in Elegy.
            run_eagerly: Settable attribute indicating whether the model should run eagerly.
                Running eagerly means that your model will be run step by step, like Python code, instead of
                using Jax's `jit` to. Your model might run slower, but it should become easier for you to debug
                it by stepping into individual layer calls.
        """
        if "rng" in kwargs and not isinstance(kwargs["rng"], (int, RNGSeq)):
            raise ValueError(
                f"rng must be one of the following types: int, RNGSeq. Got {kwargs['rng']}"
            )
        super().__init__(**kwargs)

        self.module = module
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer

        if loss is None:
            loss = {}

        if metrics is None:
            metrics = {}

        self.module_internal = generalize(module) if module is not None else None
        self.loss_internal = Losses(loss)
        self.metrics_internal = Metrics(metrics)
        self.optimizer_internal = (
            generalize_optimizer(optimizer) if optimizer is not None else None
        )
        self.seed = seed

    def __call__(self, *args, **kwargs):
        assert isinstance(self.states.rng, RNGSeq)
        assert self.module is not None

        return self.module.apply(
            self.states.net_params,
            self.states.net_states,
            self.states.rng,
        )(*args, **kwargs)

    def init(
        self,
        mode: Mode,
        x: tp.Any,
        y_true: tp.Any,
        sample_weight: tp.Optional[np.ndarray],
        class_weight: tp.Optional[np.ndarray],
    ) -> States:
        if self.module is None:
            raise MissingModule(
                "Trying run default `init` on a Model with no `module`, try overriding `init`."
            )

        states = States(rng=RNGSeq(self.seed))
        assert isinstance(states.rng, RNGSeq)

        x_args, x_kwargs = utils.get_input_args(
            x,
            states=states,
            training=True,
        )
        y_pred, net_params, net_states = self.module_internal.init(states.rng)(
            *x_args, **x_kwargs
        )

        states = states.update(net_states=net_states, net_params=net_params)

        if mode == Mode.pred:
            return states

        assert isinstance(states.rng, RNGSeq)
        metrics_logs, metrics_states = self.metrics_internal.init(states.rng)(
            x=x,
            y_true=y_true,
            y_pred=y_pred,
            net_params=net_params,
            net_states=net_states,
            metrics_states=UNINITIALIZED,
            sample_weight=sample_weight,
            class_weight=class_weight,
            rng=states.rng,
            training=True,
        )

        loss, loss_logs, loss_logs_states = self.loss_internal.init(states.rng)(
            x=x,
            y_true=y_true,
            y_pred=y_pred,
            net_params=net_params,
            net_states=net_states,
            metrics_states=UNINITIALIZED,
            sample_weight=sample_weight,
            class_weight=class_weight,
            rng=states.rng,
            training=True,
        )

        states = states.update(metrics_states=(metrics_states, loss_logs_states))
        assert isinstance(states.rng, RNGSeq)

        if mode == Mode.test:
            return states

        if self.optimizer is not None:
            optimizer_states = self.optimizer_internal.init(states.rng, net_params)
        else:
            optimizer_states = None

        states = states.update(optimizer_states=optimizer_states)

        return states

    def pred_step(
        self,
        # net_params: tp.Any,
        x: tp.Any,
        # net_states: tp.Any,
        # rng: RNG,
        training: bool,
        states: States,
    ) -> Prediction:

        if self.module is None:
            raise MissingModule(
                "Trying run default `pred_step` on a Model with no `module`, try overriding `pred_step`."
            )

        x_args, x_kwargs = utils.get_input_args(
            x,
            states=states,
            training=training,
        )

        assert isinstance(states.rng, RNGSeq)

        y_pred, net_params, net_states = self.module_internal.apply(
            states.net_params, states.net_states, states.rng
        )(*x_args, **x_kwargs)

        return Prediction(
            pred=y_pred,
            states=states.update(net_states=net_states, net_params=net_params),
        )

    def test_step(
        self,
        # net_params: tp.Any,
        x: tp.Any,
        y_true: tp.Any,
        # net_states: tp.Any,
        # metrics_states: tp.Any,
        sample_weight: tp.Optional[np.ndarray],
        class_weight: tp.Optional[np.ndarray],
        # rng: RNG,
        training: bool,
        states: States,
    ) -> Evaluation:

        # TODO: add DI
        y_pred, states = self.pred_step_internal(
            states=states,
            x=x,
            training=training,
        )
        assert isinstance(states.rng, RNGSeq)

        metrics_states, loss_states = states.metrics_states

        metrics_logs, metrics_states = self.metrics_internal.apply(
            states=metrics_states, rng=states.rng
        )(
            x=x,
            y_true=y_true,
            y_pred=y_pred,
            net_params=states.net_params,
            net_states=states.net_states,
            metrics_states=states.metrics_states,
            sample_weight=sample_weight,
            class_weight=class_weight,
            rng=states.rng,
            training=False,
            states=states,
        )

        loss, loss_logs, loss_states = self.loss_internal.apply(states=loss_states)(
            x=x,
            y_true=y_true,
            y_pred=y_pred,
            net_params=states.net_params,
            net_states=states.net_states,
            metrics_states=states.metrics_states,
            sample_weight=sample_weight,
            class_weight=class_weight,
            rng=states.rng,
            training=False,
            states=states,
        )

        logs = utils.merge_with_unique_names(metrics_logs, loss_logs)
        states = states.update(metrics_states=(metrics_states, loss_states))

        return Evaluation(loss, logs, states)

    def grad_step(
        self,
        x: tp.Any,
        y_true: tp.Any,
        sample_weight: tp.Optional[np.ndarray],
        class_weight: tp.Optional[np.ndarray],
        training: bool,
        states: States,
    ) -> Backprop:
        def loss_fn(
            net_params: tp.Any,
            states: States,
            x: tp.Any,
            y_true: tp.Any,
            sample_weight: tp.Optional[np.ndarray],
            class_weight: tp.Optional[np.ndarray],
        ):
            states = states.update(net_params=net_params)
            loss, logs, states = self.test_step_internal(
                states=states,
                x=x,
                y_true=y_true,
                sample_weight=sample_weight,
                class_weight=class_weight,
                training=training,
            )

            return loss, (logs, states)

        (loss, (logs, states)), grads = hooks.value_and_grad(loss_fn, has_aux=True)(
            states.net_params,
            states,
            x,
            y_true,
            sample_weight,
            class_weight,
        )

        return Backprop(loss, logs, states, grads)

    def train_step(
        self,
        # net_params: tp.Any,
        x: tp.Any,
        y_true: tp.Any,
        # net_states: tp.Any,
        # metrics_states: tp.Any,
        # optimizer_states: tp.Any,
        sample_weight: tp.Optional[np.ndarray],
        class_weight: tp.Optional[np.ndarray],
        # rng: RNG,
        training: bool,
        states: States,
    ) -> Training:

        if self.optimizer is None:
            raise MissingOptimizer(
                "Trying to run `train_step` without an optimizer, "
                "please provide an optimizer to the Model(...) constructor or "
                "override `train_step`."
            )
        elif isinstance(self.states.optimizer_states, Uninitialized):
            raise ValueError(
                f"Trying to run default `train_step` with an optimizer "
                "but `optimizer_states` was not initialized on `init`. Please initialize optimizer."
            )

        loss, logs, states, grads = self.grad_step_internal(
            states=states,
            x=x,
            y_true=y_true,
            sample_weight=sample_weight,
            class_weight=class_weight,
            training=training,
        )

        assert isinstance(states.rng, RNGSeq)

        # calculate current lr before update
        if isinstance(self.optimizer, Optimizer):
            lr = self.optimizer.current_lr(self.states.optimizer_states)

            if lr is not None:
                logs["lr"] = lr

        net_params, optimizer_states = self.optimizer_internal.apply(
            states.net_params, grads, states.optimizer_states, states.rng
        )

        states = states.update(net_params=net_params, optimizer_states=optimizer_states)

        return Training(logs, states)


class Metrics:
    metrics: tp.Dict[str, GeneralizedModule]

    def __init__(self, modules: tp.Any):
        names: tp.Set[str] = set()

        def get_name(module, path):
            name = utils.get_name(module)
            return f"{path}/{name}" if path else name

        self.metrics = {
            utils.get_unique_name(names, get_name(module, path)): generalize(
                module,
                callable_default=AvgMetric,
            )
            for path, module in utils.flatten_names(modules)
        }

    def calculate_metrics(
        self, callback: tp.Callable[[str, GeneralizedModule], OutputStates]
    ) -> tp.Tuple[Logs, tp.Any]:

        states = {}
        logs = hooks.get_metrics()

        if logs is None:
            logs = {}

        for name, module in self.metrics.items():
            y_pred, _, states[name] = callback(name, module)

            names = set()
            for inner_name, inner_value in utils.flatten_names(y_pred):
                inner_name = f"{name}/{inner_name}" if inner_name else name
                inner_name = utils.get_unique_name(names, inner_name)

                logs[inner_name] = inner_value

        return logs, states

    def init(self, rng: RNGSeq) -> tp.Callable[..., tp.Tuple[Logs, tp.Any]]:
        def lambda_(*args, **kwargs):
            def callback(name, module):
                kwargs_ = kwargs.copy()
                kwargs_["metrics_states"] = None
                return module.init(rng)(*args, **kwargs_)

            return self.calculate_metrics(callback)

        return lambda_

    def apply(
        self, states: tp.Any, rng: RNGSeq
    ) -> tp.Callable[..., tp.Tuple[Logs, tp.Any]]:
        assert states is not None

        def lambda_(*args, **kwargs):
            def callback(name, module):
                kwargs_ = kwargs.copy()
                kwargs_["metrics_states"] = states[name]

                return module.apply(None, states[name], rng)(*args, **kwargs_)

            return self.calculate_metrics(callback)

        return lambda_


class AvgMetric(GeneralizedModule):
    def __init__(self, f: tp.Callable):
        self.f = f

    def init(self, rng: RNGSeq) -> tp.Callable[..., OutputStates]:
        def _lambda(*args, **kwargs) -> OutputStates:

            preds = utils.inject_dependencies(self.f)(*args, **kwargs)

            if isinstance(preds, OutputStates):
                return preds

            n = 0
            total = jax.tree_map(lambda x: jnp.zeros_like(x), preds)
            return OutputStates(
                preds=preds,
                params=UNINITIALIZED,
                states=(n, total),
            )

        return _lambda

    def apply(
        self,
        params: tp.Any,
        states: tp.Any,
        rng: RNGSeq,
    ) -> tp.Callable[..., OutputStates]:
        def _lambda(*args, **kwargs) -> OutputStates:

            preds = utils.inject_dependencies(self.f)(*args, **kwargs)

            if isinstance(preds, OutputStates):
                return preds

            n, total = states
            n += 1
            total = jax.tree_multimap(lambda a, b: a + b, preds, total)
            preds = jax.tree_map(lambda total: total / n, total)
            return OutputStates(
                preds=preds,
                params=UNINITIALIZED,
                states=(n, total),
            )

        return _lambda


class Losses:
    losses: tp.Dict[str, tp.Callable]
    loss_metrics: "LossMetrics"

    def __init__(self, losses: tp.Any):
        names: tp.Set[str] = set()

        def get_name(loss_fn, path):
            name = utils.get_name(loss_fn)
            return f"{path}/{name}" if path else name

        self.losses = {
            utils.get_unique_name(names, get_name(loss_fn, path)): loss_fn
            for path, loss_fn in utils.flatten_names(losses)
        }
        self.loss_metrics = LossMetrics()

    def calculate_losses(self, *args, **kwargs) -> Logs:
        logs: Logs = {}

        for name, loss_fn in self.losses.items():
            losses = utils.inject_dependencies(loss_fn)(*args, **kwargs)

            names = set()
            for inner_name, loss in utils.flatten_names(losses):
                inner_name = f"{name}/{inner_name}" if inner_name else name
                inner_name = utils.get_unique_name(names, inner_name)

                logs[inner_name] = loss

        return logs

    def init(self, rng: RNGSeq) -> tp.Callable[..., tp.Tuple[Scalar, Logs, tp.Any]]:
        def _lambda(*args, **kwargs):
            module_logs = self.calculate_losses(*args, **kwargs)
            hooks_logs = hooks.get_losses()

            if hooks_logs is None:
                hooks_logs = {}

            loss = sum(hooks_logs.values(), 0.0) + sum(module_logs.values(), 0.0)
            loss_logs = dict(loss=loss)

            logs = utils.merge_with_unique_names(loss_logs, hooks_logs, module_logs)

            names = set()
            logs = {
                utils.get_unique_name(names, f"{name}_loss")
                if "loss" not in name
                else utils.get_unique_name(names, name): value
                for name, value in logs.items()
            }

            logs, states = self.loss_metrics.init(rng=rng)(logs)

            return loss, logs, states

        return _lambda

    def apply(self, states: tp.Any) -> tp.Callable[..., tp.Tuple[Scalar, Logs, tp.Any]]:
        def _lambda(*args, **kwargs):
            module_logs = self.calculate_losses(*args, **kwargs)
            hooks_logs = hooks.get_losses()

            if hooks_logs is None:
                hooks_logs = {}

            loss = sum(hooks_logs.values(), 0.0) + sum(module_logs.values(), 0.0)
            loss_logs = dict(loss=loss)

            logs = utils.merge_with_unique_names(loss_logs, hooks_logs, module_logs)

            names = set()
            logs = {
                utils.get_unique_name(names, f"{name}_loss")
                if "loss" not in name
                else utils.get_unique_name(names, name): value
                for name, value in logs.items()
            }

            logs, states_ = self.loss_metrics.apply(states)(logs)

            return loss, logs, states_

        return _lambda


class LossMetrics(module.Module):
    def call(self, logs):

        count = self.add_parameter("count", lambda: jnp.array(0, dtype=jnp.int32))
        total = self.add_parameter("total", lambda: jax.tree_map(jnp.zeros_like, logs))

        count = count + 1
        total = jax.tree_multimap(lambda a, b: a + b, total, logs)

        self.update_parameter("count", count)
        self.update_parameter("total", total)

        return jax.tree_map(lambda total: total / count, total)
