import typing as tp
from io import StringIO

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from elegy import hooks, module, types, utils
from elegy.model import model_core
from elegy.model.generalized_module.generalized_module import (
    GeneralizedModule,
    generalize,
    is_generalizable,
)
from elegy.model.generalized_optimizer.generalized_optimizer import (
    GeneralizedOptimizer,
    generalize_optimizer,
)
from elegy.model.model_base import ModelBase
from elegy.model.model_core import model_context
from elegy.optimizer import Optimizer
from jax._src.random import t
from tabulate import tabulate


class Model(ModelBase):
    module: tp.Any = None
    loss: tp.Any = None
    metrics: tp.Any = None
    optimizer: tp.Any = None
    seed: int = 42

    api_module: tp.Optional[GeneralizedModule]
    api_loss: "Losses"
    api_metrics: "Metrics"
    api_optimizer: tp.Optional[GeneralizedOptimizer]

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
        if "rng" in kwargs and not isinstance(kwargs["rng"], (int, types.RNGSeq)):
            raise ValueError(
                f"rng must be one of the following types: int, types.RNGSeq. Got {kwargs['rng']}"
            )

        super().__init__(rng=types.RNGSeq(seed), **kwargs)

        self.module = module
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer

        if loss is None:
            loss = {}

        if metrics is None:
            metrics = {}

        self.api_module = generalize(module) if module is not None else None
        self.api_loss = Losses(loss)
        self.api_metrics = Metrics(metrics)
        self.api_optimizer = (
            generalize_optimizer(optimizer) if optimizer is not None else None
        )
        self.seed = seed

    def __call__(self, *args, **kwargs):
        assert isinstance(self.states.rng, types.RNGSeq)
        assert self.module is not None

        return self.module.apply(
            self.states.net_params,
            self.states.net_states,
            self.states.rng,
        )(*args, **kwargs)

    def update_modules(self):
        if self.api_module is not None:
            net_params, net_states = self.api_module.update(
                params=self.states.net_params,
                states=self.states.net_states,
            )
            self.states = self.states.update(
                net_params=net_params, net_states=net_states
            )

    def pred_step(
        self,
        # net_params: tp.Any,
        x: tp.Any,
        # net_states: tp.Any,
        # rng: types.RNG,
        training: bool,
        initializing: bool,
        states: types.States,
    ) -> model_core.PredStep:

        if self.module is None:
            raise types.MissingModule(
                "Trying run default `pred_step` on a Model with no `module`, try overriding `pred_step`."
            )

        # [DI]
        x_args, x_kwargs = utils.get_input_args(
            x,
            states=states,
            training=training,
            initializing=initializing,
        )

        assert isinstance(states.rng, types.RNGSeq)

        if initializing:
            module_fn = self.api_module.init(states.rng)
        else:
            module_fn = self.api_module.apply(
                states.net_params,
                states.net_states,
                states.rng,
            )

        y_pred, net_params, net_states = module_fn(*x_args, **x_kwargs)

        return model_core.PredStep.simple(
            y_pred=y_pred,
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
        # rng: types.RNG,
        training: bool,
        initializing: bool,
        states: types.States,
    ) -> model_core.TestStep:

        # TODO: add DI
        y_pred, states, aux_losses, aux_metrics, _ = self.call_pred_step(
            states=states,
            x=x,
            training=training,
            initializing=initializing,
        )
        assert isinstance(states.rng, types.RNGSeq)

        if initializing:
            metrics_states, loss_states = None, None
            metrics_fn = self.api_metrics.init(aux_metrics, states.rng)
            losses_fn = self.api_loss.init(aux_losses, states.rng)
        else:
            metrics_states, loss_states = states.metrics_states
            metrics_fn = self.api_metrics.apply(
                aux_metrics,
                states.rng,
                metrics_states,
            )
            losses_fn = self.api_loss.apply(aux_losses, loss_states)

        # [DI]
        metrics_logs, metrics_states = metrics_fn(
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
            initializing=initializing,
            states=states,
        )

        # [DI]
        loss, loss_logs, loss_states = losses_fn(
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
            initializing=initializing,
            states=states,
        )

        logs = utils.merge_with_unique_names(metrics_logs, loss_logs)
        states = states.update(metrics_states=(metrics_states, loss_states))

        return model_core.TestStep(loss, logs, states)

    def grad_step(
        self,
        x: tp.Any,
        y_true: tp.Any,
        sample_weight: tp.Optional[np.ndarray],
        class_weight: tp.Optional[np.ndarray],
        states: types.States,
        training: bool,
        initializing: bool,
    ) -> model_core.GradStep:
        def loss_fn(
            net_params: tp.Any,
            states: types.States,
            x: tp.Any,
            y_true: tp.Any,
            sample_weight: tp.Optional[np.ndarray],
            class_weight: tp.Optional[np.ndarray],
        ):
            states = states.update(net_params=net_params)
            loss, logs, states = self.call_test_step(
                states=states,
                x=x,
                y_true=y_true,
                sample_weight=sample_weight,
                class_weight=class_weight,
                training=training,
                initializing=initializing,
            )

            return loss, (logs, states)

        (loss, (logs, states)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            states.net_params,
            states,
            x,
            y_true,
            sample_weight,
            class_weight,
        )

        return model_core.GradStep(loss, logs, states, grads)

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
        # rng: types.RNG,
        states: types.States,
        training: bool,
        initializing: bool,
    ) -> model_core.TrainStep:

        if initializing:
            loss, logs, states = self.call_test_step(
                states=states,
                x=x,
                y_true=y_true,
                sample_weight=sample_weight,
                class_weight=class_weight,
                training=training,
                initializing=initializing,
            )
            grads = None

        else:
            loss, logs, states, grads = self.call_grad_step(
                states=states,
                x=x,
                y_true=y_true,
                sample_weight=sample_weight,
                class_weight=class_weight,
                training=training,
                initializing=initializing,
            )

        if self.optimizer is None:
            raise types.MissingOptimizer(
                "Trying to run `train_step` without an optimizer, "
                "please provide an optimizer to the Model(...) constructor or "
                "override `train_step`."
            )
        assert isinstance(states.rng, types.RNGSeq)

        # calculate current lr before update
        if initializing:
            optimizer_states = self.api_optimizer.init(states.rng, states.net_params)
            net_params = states.net_params
        else:
            if isinstance(self.states.optimizer_states, types.Uninitialized):
                raise ValueError(
                    f"Trying to run default `train_step` with an optimizer "
                    "but `optimizer_states` was not initialized on `init`. Please initialize optimizer."
                )
            assert grads is not None

            if isinstance(self.optimizer, Optimizer):
                lr = self.optimizer.current_lr(self.states.optimizer_states)

                if lr is not None:
                    logs["lr"] = lr

            net_params, optimizer_states = self.api_optimizer.apply(
                states.net_params,
                grads,
                states.optimizer_states,
                states.rng,
            )

        states = states.update(
            net_params=net_params,
            optimizer_states=optimizer_states,
        )

        return model_core.TrainStep(logs, states)

    def summary(
        self, x, depth: int = 2, tablefmt: str = "fancy_grid", **tablulate_kwargs
    ) -> str:
        """
        Prints a summary of the network.
        Arguments:
            x: A sample of inputs to the network.
            depth: The level number of nested level which will be showed.
                Information about summaries from modules deeper than `depth`
                will be aggregated together.
            tablefmt: A string representing the style of the table generated by
                `tabulate`. See
                [python-tabulate](https://github.com/astanin/python-tabulate)
                for more options.
            tablulate_kwargs: Additional keyword arguments passed to `tabulate`.
                See [python-tabulate](https://github.com/astanin/python-tabulate)
                for more options.
        """
        with model_context(types.Mode.summary):
            self.maybe_initialize(x=x)

            method = (
                self.call_pred_step if self.run_eagerly else self.call_pred_step_jit
            )

            training = False
            initializing = False

            _, _, _, _, summaries = method(
                x,
                self.states,
                training,
                initializing,
            )

        assert summaries is not None

        def format_output(outputs) -> str:
            file = StringIO()
            outputs = jax.tree_map(lambda x: f"{x.shape}{{pad}}  {x.dtype}", outputs)
            yaml.safe_dump(
                outputs, file, default_flow_style=False, indent=2, explicit_end=False
            )
            return file.getvalue().replace("\n...", "")

        def format_size(size):
            return (
                f"{size / 1e9 :,.1f} GB"
                if size > 1e9
                else f"{size / 1e6 :,.1f} MB"
                if size > 1e6
                else f"{size / 1e3 :,.1f} KB"
                if size > 1e3
                else f"{size:,} B"
            )

        table: tp.List = [["Inputs", format_output(x), "", ""]]

        for path, module, value in summaries:

            module_depth = len(path)

            if module_depth > depth:
                continue

            include_submodules = module_depth == depth

            module_params, module_states = self.api_module.get_summary_params(
                path=path,
                module=module,
                value=value,
                include_submodules=include_submodules,
                net_params=self.states.net_params,
                net_states=self.states.net_states,
            )

            params_count = (
                utils.parameters_count(module_params)
                if module_params is not None
                else 0
            )
            params_size = (
                utils.parameters_bytes(module_params)
                if module_params is not None
                else 0
            )
            states_count = (
                utils.parameters_count(module_states)
                if module_states is not None
                else 0
            )
            states_size = (
                utils.parameters_bytes(module_states)
                if module_states is not None
                else 0
            )
            base_name = "/".join(map(str, path))

            if not base_name:
                base_name = "*"

            class_name = module.__class__.__name__ if is_generalizable(module) else ""

            table.append(
                [
                    f"{base_name}{{pad}}  {class_name}",
                    format_output(value),
                    f"{params_count:,}{{pad}}    {format_size(params_size)}"
                    if params_count > 0
                    else "",
                    f"{states_count:,}{{pad}}    {format_size(states_size)}"
                    if states_count > 0
                    else "",
                ]
            )

        # add padding
        for col in range(4):
            max_length = max(
                len(line.split("{pad}")[0])
                for row in table
                for line in row[col].split("\n")
            )

            for row in table:
                row[col] = "\n".join(
                    line.format(
                        pad=" " * (max_length - len(line.rstrip().split("{pad}")[0]))
                    )
                    for line in row[col].rstrip().split("\n")
                )

        # global summaries
        params_count = utils.parameters_count(self.states.net_params)
        params_size = utils.parameters_bytes(self.states.net_params)
        states_count = utils.parameters_count(self.states.net_states)
        states_size = utils.parameters_bytes(self.states.net_states)
        total_count = params_count + states_count
        total_size = params_size + states_size

        summary = (
            "\n"
            + tabulate(
                table,
                headers=[
                    "Layer",
                    "Outputs Shape",
                    "Trainable\nParameters",
                    "Non-trainable\nParameters",
                ],
                tablefmt=tablefmt,
                **tablulate_kwargs,
            )
            + "\n"
            + tabulate(
                [
                    [
                        f"Total Parameters:",
                        f"{total_count:,}",
                        f"{format_size(total_size)}" if total_count > 0 else "",
                    ],
                    [
                        f"Trainable Parameters:",
                        f"{params_count:,}",
                        f"{format_size(params_size)}" if params_count > 0 else "",
                    ],
                    [
                        f"Non-trainable Parameters:",
                        f"{states_count:,}",
                        f"{format_size(states_size)}" if states_count > 0 else "",
                    ],
                ],
                tablefmt="plain",
            )
            + "\n"
        )

        print(summary)
        return summary


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
        self,
        aux_metrics: types.Logs,
        callback: tp.Callable[[str, GeneralizedModule], types.OutputStates],
    ) -> tp.Tuple[types.Logs, tp.Any]:

        states = {}

        for name, module in self.metrics.items():
            y_pred, _, states[name] = callback(name, module)

            names = set()
            for inner_name, inner_value in utils.flatten_names(y_pred):
                inner_name = f"{name}/{inner_name}" if inner_name else name
                inner_name = utils.get_unique_name(names, inner_name)

                aux_metrics[inner_name] = inner_value

        return aux_metrics, states

    def init(
        self,
        aux_metrics: types.Logs,
        rng: types.RNGSeq,
    ) -> tp.Callable[..., tp.Tuple[types.Logs, tp.Any]]:
        def lambda_(*args, **kwargs):
            def callback(name, module):
                kwargs_ = kwargs.copy()
                kwargs_["metrics_states"] = None
                return module.init(rng)(*args, **kwargs_)

            return self.calculate_metrics(aux_metrics, callback)

        return lambda_

    def apply(
        self,
        aux_metrics: types.Logs,
        rng: types.RNGSeq,
        states: tp.Any,
    ) -> tp.Callable[..., tp.Tuple[types.Logs, tp.Any]]:
        assert states is not None

        def lambda_(*args, **kwargs):
            def callback(name, module):
                kwargs_ = kwargs.copy()
                kwargs_["metrics_states"] = states[name]

                return module.apply(None, states[name], rng)(*args, **kwargs_)

            return self.calculate_metrics(aux_metrics, callback)

        return lambda_


class AvgMetric(GeneralizedModule):
    def __init__(self, f: tp.Callable):
        self.f = f

    def init(self, rng: types.RNGSeq) -> tp.Callable[..., types.OutputStates]:
        def _lambda(*args, **kwargs) -> types.OutputStates:

            preds = utils.inject_dependencies(self.f)(*args, **kwargs)

            if isinstance(preds, types.OutputStates):
                return preds

            n = 0
            total = jax.tree_map(lambda x: jnp.zeros_like(x), preds)
            return types.OutputStates(
                preds=preds,
                params=types.UNINITIALIZED,
                states=(n, total),
            )

        return _lambda

    def apply(
        self,
        params: tp.Any,
        states: tp.Any,
        rng: types.RNGSeq,
    ) -> tp.Callable[..., types.OutputStates]:
        def _lambda(*args, **kwargs) -> types.OutputStates:

            preds = utils.inject_dependencies(self.f)(*args, **kwargs)

            if isinstance(preds, types.OutputStates):
                return preds

            n, total = states
            n += 1
            total = jax.tree_multimap(lambda a, b: a + b, preds, total)
            preds = jax.tree_map(lambda total: total / n, total)
            return types.OutputStates(
                preds=preds,
                params=types.UNINITIALIZED,
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

    def calculate_losses(self, *args, **kwargs) -> types.Logs:
        logs: types.Logs = {}

        for name, loss_fn in self.losses.items():
            losses = utils.inject_dependencies(loss_fn)(*args, **kwargs)

            names = set()
            for inner_name, loss in utils.flatten_names(losses):
                inner_name = f"{name}/{inner_name}" if inner_name else name
                inner_name = utils.get_unique_name(names, inner_name)

                logs[inner_name] = loss

        return logs

    def init(
        self,
        aux_losses: types.Logs,
        rng: types.RNGSeq,
    ) -> tp.Callable[..., tp.Tuple[types.Scalar, types.Logs, tp.Any]]:
        def _lambda(*args, **kwargs):
            module_logs = self.calculate_losses(*args, **kwargs)

            loss = sum(aux_losses.values(), 0.0) + sum(module_logs.values(), 0.0)
            loss_logs = dict(loss=loss)

            logs = utils.merge_with_unique_names(loss_logs, aux_losses, module_logs)

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

    def apply(
        self,
        aux_losses: types.Logs,
        states: tp.Any,
    ) -> tp.Callable[..., tp.Tuple[types.Scalar, types.Logs, tp.Any]]:
        def _lambda(*args, **kwargs):
            module_logs = self.calculate_losses(*args, **kwargs)

            loss = sum(aux_losses.values(), 0.0) + sum(module_logs.values(), 0.0)
            loss_logs = dict(loss=loss)

            logs = utils.merge_with_unique_names(loss_logs, aux_losses, module_logs)

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
