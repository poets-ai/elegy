import typing as tp
from io import StringIO

import flax
import jax
import jax.numpy as jnp
import numpy as np
import treex as tx
from optax import GradientTransformation
from treex.nn.haiku_module import HaikuModule

from elegy import types, utils
from elegy.model.model_base import ModelBase
from elegy.model.model_core import GradStep, PredStep, TestStep, TrainStep

M = tp.TypeVar("M", bound="Model")
U = tp.TypeVar("U", bound="tx.Module")

try:
    import haiku as hk

    TransformedWithState = hk.TransformedWithState
except ImportError:
    hk = None
    TransformedWithState = type(None)


class Model(tp.Generic[U], ModelBase):
    """
    Model provides an Estimator-like API similar to Keras.
    """

    # pytree
    module: tp.Optional[U] = tx.node()
    loss_and_logs: tp.Optional[tx.LossAndLogs]
    optimizer: tp.Optional[tx.Optimizer]

    # static
    seed: int = 42

    @tp.overload
    def __init__(
        self: "Model[tx.FlaxModule]",
        module: flax.linen.module.Module,
        loss: tp.Any = None,
        metrics: tp.Any = None,
        optimizer: tp.Optional[tp.Union[tx.Optimizer, GradientTransformation]] = None,
        seed: int = 42,
        eager: bool = False,
    ):
        ...

    if hk is not None:

        @tp.overload
        def __init__(
            self: "Model[tx.HaikuModule]",
            module: hk.TransformedWithState,
            loss: tp.Any = None,
            metrics: tp.Any = None,
            optimizer: tp.Optional[
                tp.Union[tx.Optimizer, GradientTransformation]
            ] = None,
            seed: int = 42,
            eager: bool = False,
        ):
            ...

    @tp.overload
    def __init__(
        self: "Model[U]",
        module: U,
        loss: tp.Any = None,
        metrics: tp.Any = None,
        optimizer: tp.Optional[tp.Union[tx.Optimizer, GradientTransformation]] = None,
        seed: int = 42,
        eager: bool = False,
    ):
        ...

    def __init__(
        self,
        module: tp.Union[
            flax.linen.module.Module, TransformedWithState, tx.Module, None
        ] = None,
        loss: tp.Any = None,
        metrics: tp.Any = None,
        optimizer: tp.Optional[tp.Union[tx.Optimizer, GradientTransformation]] = None,
        seed: int = 42,
        eager: bool = False,
    ):
        """
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
            eager: Settable attribute indicating whether the model should run eagerly.
                Running eagerly means that your model will be run step by step, like Python code, instead of
                using Jax's `jit` to. Your model might run slower, but it should become easier for you to debug
                it by stepping into individual layer calls.
        """
        super().__init__(seed=seed, eager=eager)

        if isinstance(module, flax.linen.module.Module):
            self.module = tx.FlaxModule(module)

        elif TransformedWithState is not None and isinstance(
            module, TransformedWithState
        ):
            self.module = tx.HaikuModule(module)
        else:
            self.module = module

        self.optimizer = (
            tx.Optimizer(optimizer)
            if isinstance(optimizer, GradientTransformation)
            else optimizer
        )
        self.loss_and_logs = None

        self._losses_and_metrics = lambda: (
            loss,
            metrics,
        )

    def __call__(self, *args, **kwargs) -> tp.Any:
        assert self.module is not None

        return self.module(*args, **kwargs)

    # ----------------------------------------------------------------
    # implement low-level API methods
    # ----------------------------------------------------------------

    def init_step(
        self: M,
        key: jnp.ndarray,
        inputs: tp.Any,
    ) -> M:

        if self.module is not None:
            self.module = self.module.init(key, inputs=inputs)

        if self.optimizer is not None:
            params = self.parameters()
            self.optimizer = self.optimizer.init(params)

        losses, metrics = self._losses_and_metrics()
        aux_losses = self.loss_logs()
        aux_metrics = self.metric_logs()

        self.loss_and_logs = tx.LossAndLogs(
            losses=losses,
            metrics=metrics,
            aux_losses=aux_losses,
            aux_metrics=aux_metrics,
        )

        return self

    def pred_step(
        self: M,
        inputs: tp.Any,
    ) -> PredStep[M]:
        model: M = self

        if model.module is None:
            raise types.MissingModule(
                "Trying to run default `pred_step` on a Model with no `module`, try overriding `pred_step` or set `module`"
            )

        inputs_obj = tx.Inputs.from_value(inputs)

        preds = model.module(*inputs_obj.args, **inputs_obj.kwargs)

        return preds, model

    def test_step(
        self: M,
        inputs: tp.Any,
        labels: tp.Mapping[str, tp.Any],
    ) -> TestStep[M]:
        model: M = self

        if model.module is None:
            raise types.MissingModule(
                "Trying to run default `test_step` on a Model with no `loss_and_logs`, try overriding `test_step` or set `loss_and_logs`"
            )

        preds, model = model.pred_step(inputs)
        assert model.loss_and_logs is not None

        aux_losses = model.loss_logs()
        aux_metrics = model.metric_logs()

        extended_labels = {
            "inputs": inputs,
            "preds": preds,
            "model": model,
            "parameters": model.parameters(),
            "batch_stats": model.batch_stats(),
            "rngs": model.rngs(),
            "model_states": model.model_states(),
            "states": model.states(),
            "metric_logs": model.metric_logs(),
            "loss_logs": model.loss_logs(),
            "logs": model.logs(),
            **labels,
        }

        losses_kwargs = extended_labels
        metrics_kwargs = extended_labels

        loss, losses_logs, metrics_logs = model.loss_and_logs.batch_loss_epoch_logs(
            **losses_kwargs,
            metrics_kwargs=metrics_kwargs,
            aux_losses=aux_losses,
            aux_metrics=aux_metrics,
        )

        logs = {**losses_logs, **metrics_logs}

        return loss, logs, model

    @staticmethod
    def loss_fn(
        params: M,
        model: M,
        inputs: tp.Any,
        labels: tp.Mapping[str, tp.Any],
    ) -> tp.Tuple[jnp.ndarray, tp.Tuple[types.Logs, M]]:
        model = model.merge(params)
        loss, logs, model = model.test_step(inputs, labels)
        return loss, (logs, model)

    @staticmethod
    def _is_trainable(field_info: tx.FieldInfo) -> bool:
        if isinstance(field_info.module, tx.Module):
            return not field_info.module.frozen and field_info.module.training
        else:
            return True

    def train_step(
        self: M,
        inputs: tp.Any,
        labels: tp.Mapping[str, tp.Any],
    ) -> TrainStep[M]:
        model: M = self
        grads: M
        logs: types.Logs
        model: M = self

        if model.optimizer is None:
            raise types.MissingModule(
                "Trying to run default `train_step` on a Model with no `optimizer`, try overriding `train_step` or set `optimizer`"
            )

        params = model.parameters(self._is_trainable)

        grad_fn = jax.grad(self.loss_fn, has_aux=True)
        grads, (logs, model) = grad_fn(params, model, inputs, labels)

        assert model.optimizer is not None

        params = model.optimizer.update(grads, params)
        model = model.merge(params)

        return logs, model

    def reset_metrics(self) -> None:
        if self.loss_and_logs is not None:
            self.loss_and_logs.reset()

    # ----------------------------------------------------------------
    # Model-only methods
    # ----------------------------------------------------------------

    def summary(
        self,
        inputs: tp.Any = tx.MISSING,
        depth: int = 2,
        return_repr: bool = False,
    ) -> tp.Optional[str]:
        """
        Prints a summary of the network. The representation is module dependent,
        if a library provides a representation, it will be used, else elegy will
        define its own.

        Arguments:
            x: A sample of inputs to the network.
            depth: The level number of nested level which will be showed.
                Information about summaries from modules deeper than `depth`
                will be aggregated together.
            return_repr: If True, the summary will be returned as a string and will not be printed.
            eval_shape: If True, jax.eval_shape is used to calculate all shapes, this avoids actually
                running the computation as only shapes are calculated (turn off if trying to debug).
        """
        assert self.module is not None

        if not self.initialized:
            self.init_on_batch(inputs)

        summary = self.module.tabulate(
            inputs=inputs,
            depth=depth,
        )

        if return_repr:
            return summary
        else:
            print(summary)
