import typing as tp
from io import StringIO

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from elegy import hooks, module, types, utils
from elegy.model import model_core
from elegy.generalized_module.generalized_module import (
    GeneralizedModule,
    generalize,
    is_generalizable,
)
from elegy.generalized_optimizer.generalized_optimizer import (
    GeneralizedOptimizer,
    generalize_optimizer,
)
from elegy.model.model_base import ModelBase
from elegy.optimizer import Optimizer
from tabulate import tabulate


class Model(ModelBase):
    """
    Model is a framework-agnostic trainer interface that is tasked with performing training, evaluation, and inference.
    It provides 2 main APIs:

    * The high-level API provides the Keras-like experience of simplicity and speed. The user provides things like the
        a modules, losses, metrics, and callbacks and Elegy takes care of the rest.
    * The low-level API provides the Pytorch Lightning-like experience of experisiveness and power. The user can
        override methods like `pred_step`, `test_step`, and `train_step` to perform jax-based computation with very
        few restrictions.

    ## High-level API
    To use the high-level API you first have to define a network architecture in a Module, currently we support Modules from
    Flax, Haiku, Elegy, and pure jax functions. Using `elegy.Module` this could look like this:

    ```python
    >>> import elegy, jax, optax
    >>> import jax.numpy as jnp

    >>> class MLP(elegy.Module):
    ...     def call(self, x: jnp.ndarray) -> jnp.ndarray:
    ...         x = elegy.nn.Linear(5)(x)
    ...         x = jax.nn.relu(x)
    ...         x = elegy.nn.Linear(2)(x)
    ...         return x

    ```


    You can pass this architecture to the Model API along with losses, metrics, optimizer, etc:
    ```python
    >>> model = elegy.Model(
    ...     module=MLP(),
    ...     loss=[
    ...         elegy.losses.SparseCategoricalCrossentropy(from_logits=True),
    ...         elegy.regularizers.GlobalL2(l=1e-5),
    ...     ],
    ...     metrics=elegy.metrics.SparseCategoricalAccuracy(),
    ...     optimizer=optax.rmsprop(1e-3),
    ... )

    ```

    Once the model is created, you can train the model with `model.fit()`, or use the model
    to do prediction with `model.predict()`.
    Checkout [Getting Started](https://poets-ai.github.io/elegy/getting-started) for
    additional details.

    ```python
    >>> x = jnp.ones(shape=[12, 10])
    >>> y = jnp.ones(shape=[12])

    >>> history = model.fit(x, y) # doctest: +SKIP

    ```

    Model supports optax optimizers as well as `elegy.Optimizer` which has a feature for definint + monitoring
    custom learning rate schedules, it is implemented on top of optax and you can use it like this:

    ```python
    >>> model = elegy.Model(
    ...     module=MLP(),
    ...     loss=elegy.losses.SparseCategoricalCrossentropy(from_logits=True),
    ...     metrics=elegy.metrics.SparseCategoricalAccuracy(),
    ...     optimizer=elegy.Optimizer(
    ...         # one or more optax optimizers as *args,
    ...         # these will be passed to optax.chain(...)
    ...         optax.adam(1.0), # <---- important to set this to 1.0
    ...
    ...         lr_schedule=lambda step, epoch: 1 / (epoch * 100 + step),
    ...         steps_per_epoch=1000,
    ...     ),
    ...     run_eagerly=True,
    ... )

    ```
    Notice how we set the learning rate parameter of the `adam` optimizer to `1.0`, this is necessary if you want the logged `lr`
    be closer to the "actual" learning rate since this feature was implemented by chaining an additional `optax.scale_by_schedule`
    at the end.

    ## High-level API

    """

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

        super().__init__(**kwargs)

        # maybe add rng if initialized
        if self.initialized and (
            not hasattr(self.states, "rng") or self.states.rng is None
        ):
            self.states = self.states.update(rng=types.RNGSeq(seed))

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
            self.api_module.update(
                params=self.states.net_params,
                states=self.states.net_states,
            )

    # ----------------------------------------------------------------
    # implement low-level API methods
    # ----------------------------------------------------------------
    def states_step(self) -> types.States:
        states = super().states_step()
        return states.update(
            rng=types.RNGSeq(self.seed),
            net_params=None,
            net_states=None,
            metrics_states=None,
            optimizer_states=None,
        )

    def init_step(
        self,
        x: tp.Any,
        y_true: tp.Any,
        sample_weight: tp.Optional[np.ndarray],
        class_weight: tp.Optional[np.ndarray],
        states: types.States,
    ) -> types.States:

        states = states.maybe_update(**self.states_step())
        training = True
        initializing = True

        _, states = utils.inject_dependencies(self.train_step)(
            x=x,
            y_true=y_true,
            sample_weight=sample_weight,
            class_weight=class_weight,
            states=states,
            initializing=initializing,
            training=training,
        )

        return states

    def summary_step(
        self,
        x: tp.Any,
        states: types.States,
    ) -> tp.List[types.SummaryTableEntry]:
        initializing = True
        training = True

        states = states.maybe_update(**self.states_step())

        with hooks.context(summaries=True):
            _, states = utils.inject_dependencies(self.pred_step)(
                x, states, initializing, training
            )
            summaries = hooks.get_summaries()

        entries: tp.List[types.SummaryTableEntry] = []

        for path, module, value in summaries:

            module_params, module_states = self.api_module.get_summary_params(
                path=path,
                module=module,
                value=value,
                net_params=states.net_params,
                net_states=states.net_states,
            )

            entries.append(
                types.SummaryTableEntry(
                    path=("/".join(map(str, path)) if path else "*"),
                    module_type_name=(
                        module.__class__.__name__ if is_generalizable(module) else ""
                    ),
                    output_value=value,
                    trainable_params_count=(
                        utils.parameters_count(module_params)
                        if module_params is not None
                        else 0
                    ),
                    trainable_params_size=(
                        utils.parameters_bytes(module_params)
                        if module_params is not None
                        else 0
                    ),
                    non_trainable_params_count=(
                        utils.parameters_count(module_states)
                        if module_states is not None
                        else 0
                    ),
                    non_trainable_params_size=(
                        utils.parameters_bytes(module_states)
                        if module_states is not None
                        else 0
                    ),
                )
            )

        entries.append(
            types.SummaryTableEntry.totals_entry(
                trainable_params_count=utils.parameters_count(states.net_params),
                trainable_params_size=utils.parameters_bytes(states.net_params),
                non_trainable_params_count=utils.parameters_count(states.net_states),
                non_trainable_params_size=utils.parameters_bytes(states.net_states),
            )
        )

        return entries

    def pred_step(
        self,
        x: tp.Any,
        states: types.States,
        initializing: bool,
        training: bool,
    ) -> model_core.PredStep:

        if self.module is None:
            raise types.MissingModule(
                "Trying run default `pred_step` on a Model with no `module`, try overriding `pred_step`."
            )

        # [DI]
        x_args, x_kwargs = utils.get_input_args(
            x,
            states=states,
            initializing=initializing,
            training=training,
        )

        assert isinstance(states.rng, types.RNGSeq)

        if initializing:
            module_fn = self.api_module.init(states.rng)
        else:
            module_fn = self.api_module.apply(
                params=states.net_params,
                states=states.net_states,
                training=training,
                rng=states.rng,
            )

        y_pred, net_params, net_states = module_fn(*x_args, **x_kwargs)

        return model_core.PredStep(
            y_pred=y_pred,
            states=states.update(net_states=net_states, net_params=net_params),
        )

    def test_step(
        self,
        x: tp.Any,
        y_true: tp.Any,
        sample_weight: tp.Optional[np.ndarray],
        class_weight: tp.Optional[np.ndarray],
        states: types.States,
        initializing: bool,
        training: bool,
    ) -> model_core.TestStep:

        with hooks.context(losses=True, summaries=True):
            y_pred, states = utils.inject_dependencies(self.pred_step)(
                x=x,
                states=states,
                initializing=initializing,
                training=training,
            )

            aux_losses = hooks.get_losses()
            aux_metrics = hooks.get_metrics()

        assert isinstance(states.rng, types.RNGSeq)

        if initializing:
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

        # [DI]metrics_states
        metrics_logs, metrics_states = metrics_fn(
            x=x,
            y_true=y_true,
            y_pred=y_pred,
            sample_weight=sample_weight,
            class_weight=class_weight,
            initializing=initializing,
            states=states,
            training=training,
        )

        # [DI]
        loss, loss_logs, loss_states = losses_fn(
            x=x,
            y_true=y_true,
            y_pred=y_pred,
            sample_weight=sample_weight,
            class_weight=class_weight,
            initializing=initializing,
            states=states,
            training=training,
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
        initializing: bool,
        training: bool,
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
            loss, logs, states = utils.inject_dependencies(self.test_step)(
                x=x,
                y_true=y_true,
                states=states,
                sample_weight=sample_weight,
                class_weight=class_weight,
                initializing=initializing,
                training=training,
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
        x: tp.Any,
        y_true: tp.Any,
        sample_weight: tp.Optional[np.ndarray],
        class_weight: tp.Optional[np.ndarray],
        states: types.States,
        initializing: bool,
        training: bool,
    ) -> model_core.TrainStep:

        if initializing:
            loss, logs, states = utils.inject_dependencies(self.test_step)(
                x=x,
                y_true=y_true,
                states=states,
                sample_weight=sample_weight,
                class_weight=class_weight,
                initializing=initializing,
                training=training,
            )
            grads = None

        else:
            loss, logs, states, grads = utils.inject_dependencies(self.grad_step)(
                x=x,
                y_true=y_true,
                states=states,
                sample_weight=sample_weight,
                class_weight=class_weight,
                initializing=initializing,
                training=training,
            )

        if initializing and self.optimizer is None:
            return model_core.TrainStep(logs, states)
        elif self.optimizer is None:
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
            assert grads is not None

            # TODO: add current_lr this to GeneralizedOptimizer and do this generically
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

    # ----------------------------------------------------------------
    # Model-only methods
    # ----------------------------------------------------------------


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
            def callback(name, module: GeneralizedModule):
                kwargs_ = kwargs.copy()
                kwargs_["metrics_states"] = states[name]

                return module.apply(
                    params=None, states=states[name], training=True, rng=rng
                )(*args, **kwargs_)

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
                params=None,
                states=(n, total),
            )

        return _lambda

    def apply(
        self,
        params: tp.Any,
        states: tp.Any,
        training: bool,
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
                params=None,
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

            logs, _, states = self.loss_metrics.init(rng=rng)(logs)

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

            parameters = None
            logs, _, states_ = self.loss_metrics.apply(parameters, states)(logs)

            return loss, logs, states_

        return _lambda


class LossMetrics(module.Module):
    def call(self, logs):

        count = self.add_parameter(
            "count", lambda: jnp.array(0, dtype=jnp.int32), trainable=False
        )
        total = self.add_parameter(
            "total", lambda: jax.tree_map(jnp.zeros_like, logs), trainable=False
        )

        count = count + 1
        total = jax.tree_multimap(lambda a, b: a + b, total, logs)

        self.update_parameter("count", count)
        self.update_parameter("total", total)

        return jax.tree_map(lambda total: total / count, total)
