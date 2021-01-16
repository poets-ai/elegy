import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
import optax
from elegy import utils
from elegy.model.generalized_module.generalized_module import (
    GeneralizedModule,
    generalize,
)
from elegy.model.model_base import ModelBase
from elegy.model.model_core import Prediction, States
from elegy.types import Logs, RNG, Evaluation, OutputStates, Prediction, Scalar
from elegy.utils import Mode, RNGSeq
from flax import linen
from elegy import hooks

LossModules = tp.Union[tp.Callable, tp.List, tp.Dict, None]
MetricsModules = tp.Union[tp.Callable, tp.List, tp.Dict, None]


class Model(ModelBase):
    module: GeneralizedModule
    loss: LossModules = None
    metrics: MetricsModules = None
    optimizer: tp.Union["Optimizer", optax.GradientTransformation, None] = None
    seed: int = 42

    def __init__(
        self,
        module: tp.Any,
        loss: LossModules = None,
        metrics: MetricsModules = None,
        optimizer: tp.Union["Optimizer", optax.GradientTransformation, None] = None,
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

        self.module = generalize(module)
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.seed = seed

    def __call__(self, *args, **kwargs):
        assert isinstance(self.states.rng, RNGSeq)

        return self.module.apply(
            self.states.net_params,
            self.states.net_states,
            self.states.rng,
            args,
            kwargs,
        )

    def init(
        self,
        mode: Mode,
        x: tp.Any,
        y_true: tp.Any,
        sample_weight: tp.Optional[np.ndarray],
        class_weight: tp.Optional[np.ndarray],
    ) -> States:
        rng = RNGSeq(self.seed)

        training = mode == Mode.train

        x_args, x_kwargs = utils.get_input_args(x, training=training)
        y_pred, net_params, net_states = self.module.init(rng, x_args, x_kwargs)

        states = States(net_states=net_states, net_params=net_params, rng=rng)

        if mode == Mode.pred:
            return states

        return states

    def pred_step(
        self,
        net_params: tp.Any,
        x: tp.Any,
        net_states: tp.Any,
        rng: RNG,
    ) -> Prediction:
        assert isinstance(rng, RNGSeq)

        x_args, x_kwargs = utils.get_input_args(x, training=False)

        y_pred, net_params, net_states = self.module.apply(
            net_params,
            net_states,
            rng,
            x_args,
            x_kwargs,
        )

        return Prediction(
            pred=y_pred,
            states=States(net_states=net_states, net_params=net_params, rng=rng),
        )

    def test_step(
        self,
        net_params: tp.Any,
        x: tp.Any,
        y_true: tp.Any,
        net_states: tp.Any,
        metrics_states: tp.Any,
        sample_weight: tp.Optional[np.ndarray],
        class_weight: tp.Optional[np.ndarray],
        rng: RNG,
    ) -> Evaluation:
        assert isinstance(rng, RNGSeq)

        y_pred, states = self.pred_step(
            net_params=net_params,
            x=x,
            net_states=net_states,
            rng=rng,
        )

        logs = {}

        return Evaluation(logs, states)

    def train_step(
        self,
        net_params: tp.Any,
        x: tp.Any,
        y_true: tp.Any,
        net_states: tp.Any,
        metrics_states: tp.Any,
        optimizer_states: tp.Any,
        sample_weight: tp.Optional[np.ndarray],
        class_weight: tp.Optional[np.ndarray],
        rng: RNG,
    ) -> Evaluation:
        ...


class Metrics:
    metrics: tp.Dict[str, GeneralizedModule]

    def __init__(self, modules: tp.Any):
        names: tp.Set[str] = set()

        def get_name(module, path):
            name = utils.get_name(module)
            return f"{path}/{name}" if path else name

        self.metrics = {
            utils.get_unique_name(names, get_name(module, path)): generalize(module)
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

    def init(self, rng: utils.RNGSeq) -> tp.Callable[..., tp.Tuple[Logs, tp.Any]]:
        return lambda *args, **kwargs: self.calculate_metrics(
            lambda name, module: module.init(rng, args, kwargs)
        )

    def apply(
        self, states: tp.Any, rng: utils.RNGSeq
    ) -> tp.Callable[..., tp.Tuple[Logs, tp.Any]]:
        assert states is not None

        return lambda *args, **kwargs: self.calculate_metrics(
            lambda name, module: module.apply(None, states[name], rng, args, kwargs)
        )


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

    def init(
        self, rng: utils.RNGSeq
    ) -> tp.Callable[..., tp.Tuple[Scalar, Logs, tp.Any]]:
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

            logs, states = self.loss_metrics.init_with_output(rng.next(), logs)

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

            logs, states_ = self.loss_metrics.apply(states, logs, mutable=True)

            return loss, logs, states_

        return _lambda


class LossMetrics(linen.Module):
    @linen.compact
    def __call__(self, logs):

        initialized = self.has_variable("batch_stats", "count")

        vcount = self.variable(
            "batch_stats", "count", lambda: jnp.array(0, dtype=jnp.int32)
        )
        vtotal = self.variable(
            "batch_stats", "total", lambda: jax.tree_map(jnp.zeros_like, logs)
        )

        total = jax.tree_multimap(lambda a, b: a + b, vtotal.value, logs)
        count = vcount.value + 1

        if initialized:
            vtotal.value = total
            vcount.value = count

        return jax.tree_map(lambda total: total / count, total)


class Optimizer:
    r"""A Module that wraps around `optax` optimizers."""

    def __init__(
        self,
        *optimizer: optax.GradientTransformation,
        lr_schedule: tp.Optional[
            tp.Callable[[int, tp.Optional[np.ndarray]], np.ndarray]
        ] = None,
        steps_per_epoch: tp.Optional[int] = None,
        **kwargs,
    ):
        r"""
        Arguments:
            optimizer: An optax `GradientTransformation` object, if more than one is passed via `*args` then they are
                grouped using `optax.chain`.
            lr_schedule: A optional callable of the form `def lr_schedule(step: int, epoch: Optional[int]) -> float` that
                returns the learning rate schedule at each time step. If `steps_per_epoch` is given then epoch is calculated,
                else epoch is None.
            steps_per_epoch: The number of steps to in an epoch, needed to caculate `epoch` from `step`.
        """
        super().__init__(**kwargs)

        if len(optimizer) == 0:
            raise ValueError("Must pass atleast 1 optimizer, got 0")
        elif len(optimizer) == 1:
            optimizer = optimizer[0]
        else:
            optimizer = optax.chain(*optimizer)

        if lr_schedule is not None:
            base_schedule = lr_schedule

            def lr_schedule(step: int) -> float:
                if steps_per_epoch is not None:
                    epoch = step // steps_per_epoch
                else:
                    epoch = None

                return base_schedule(step, epoch)

            optimizer = optax.chain(
                optimizer,
                optax.scale_by_schedule(lr_schedule),
            )

        self.optax_optimizer = optimizer
        self.lr_schedule = lr_schedule

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

    def get_effective_learning_rate(self) -> tp.Optional[float]:
        """Returns the learning rate scaled by schedule(s) that will be used for the next training step"""

        if self.initialized and self.lr_schedule is not None:
            step = self.optimizer_state[-1].count
            return self.lr_schedule(step)
