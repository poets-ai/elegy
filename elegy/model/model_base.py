from dataclasses import dataclass
from io import StringIO
import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from tabulate import tabulate

from elegy import module, utils
from elegy.losses.loss import Loss
from elegy.metrics.metric import Metric
from abc import ABC, abstractmethod
from elegy.utils import Mode

from enum import Enum
from elegy import hooks


class States(tp.NamedTuple):
    net_params: tp.Any = utils.UNINITIALIZED
    net_states: tp.Any = utils.UNINITIALIZED
    metrics_states: tp.Any = utils.UNINITIALIZED
    optimizer_states: tp.Any = utils.UNINITIALIZED

    def update(
        self,
        net_params: tp.Any = utils.UNINITIALIZED,
        net_states: tp.Any = utils.UNINITIALIZED,
        metrics_states: tp.Any = utils.UNINITIALIZED,
        optimizer_states: tp.Any = utils.UNINITIALIZED,
    ) -> "States":

        updates = {}

        if not isinstance(net_params, utils.Uninitialized):
            updates["net_params"] = net_params
        if not isinstance(net_states, utils.Uninitialized):
            updates["net_states"] = net_states
        if not isinstance(metrics_states, utils.Uninitialized):
            updates["metrics_states"] = metrics_states
        if not isinstance(optimizer_states, utils.Uninitialized):
            updates["optimizer_states"] = optimizer_states

        kwargs = {field: getattr(self, field) for field in self._fields}
        kwargs.update(**updates)

        return States(**kwargs)

    def merge(self, other: "States") -> "States":
        return other.update(*other)


class ModelBase(ABC):
    states: States
    run_eagerly: bool = False

    def __init__(
        self,
        net_params: tp.Union[utils.Uninitialized, tp.Any] = utils.UNINITIALIZED,
        net_states: tp.Union[utils.Uninitialized, tp.Any] = utils.UNINITIALIZED,
        metrics_states: tp.Union[utils.Uninitialized, tp.Any] = utils.UNINITIALIZED,
        optimizer_states: tp.Union[utils.Uninitialized, tp.Any] = utils.UNINITIALIZED,
        run_eagerly: bool = False,
    ):
        self.states = States(
            net_params=net_params,
            net_states=net_states,
            metrics_states=metrics_states,
            optimizer_states=optimizer_states,
        )
        self.run_eagerly = run_eagerly

        self._jit_functions()

    def _jit_functions(self):
        self.pred_step_internal_jit = hooks.jit(self.pred_step_internal)
        self.test_step_jit = hooks.jit(self.test_step)
        self.train_step_jit = hooks.jit(self.train_step)

    @abstractmethod
    def init(
        self,
        mode: Mode,
        x: tp.Any = (),
        y: tp.Any = None,
        sample_weight: tp.Optional[jnp.ndarray] = None,
        class_weight: tp.Optional[jnp.ndarray] = None,
    ) -> States:
        ...

    @abstractmethod
    def pred_step(
        self,
        net_params: tp.Any = None,
        x: tp.Any = (),
        net_states: tp.Any = None,
    ) -> tp.Tuple[tp.Any, States]:
        ...

    @abstractmethod
    def test_step(
        self,
        net_params: tp.Any = None,
        x: tp.Any = (),
        y: tp.Any = None,
        net_states: tp.Any = None,
        metrics_states: tp.Any = None,
        sample_weight: tp.Optional[jnp.ndarray] = None,
        class_weight: tp.Optional[jnp.ndarray] = None,
    ) -> States:
        ...

    @abstractmethod
    def train_step(
        self,
        net_params: tp.Any = None,
        x: tp.Any = (),
        y: tp.Any = None,
        net_states: tp.Any = None,
        metrics_states: tp.Any = None,
        optimizer_states: tp.Any = None,
        sample_weight: tp.Optional[jnp.ndarray] = None,
        class_weight: tp.Optional[jnp.ndarray] = None,
    ) -> States:
        ...

    def get_loss_and_grad(
        self, net_params, *args
    ) -> tp.Tuple[np.ndarray, tp.Any, States]:
        def loss_fn(net_params, *args):
            metrics_states = self.test_step(net_params, *args)
            return hooks.get_total_loss(), metrics_states

        (loss, metrics_states), grads = hooks.value_and_grad(loss_fn, has_aux=True)(
            net_params, *args
        )

        assert isinstance(metrics_states, States)

        return loss, grads, metrics_states

    def maybe_initialize(
        self,
        mode: Mode,
        x: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple] = (),
        y: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple, None] = None,
        sample_weight: tp.Optional[jnp.ndarray] = None,
        class_weight: tp.Optional[jnp.ndarray] = None,
    ):

        if (
            (
                mode == Mode.pred
                and isinstance(self.states.net_params, utils.Uninitialized)
                and isinstance(self.states.net_states, utils.Uninitialized)
            )
            or (
                mode == Mode.test
                and isinstance(self.states.metrics_states, utils.Uninitialized)
            )
            or (
                mode == Mode.train
                and isinstance(self.states.optimizer_states, utils.Uninitialized)
            )
        ):
            state_updates: States = utils.inject_dependencies(self.init)(
                mode=mode,
                x=x,
                y=y,
                sample_weight=sample_weight,
                class_weight=class_weight,
            )
            self.states = self.states.merge(state_updates)

    def pred_step_internal(
        self,
        net_params: tp.Any = None,
        x: tp.Any = (),
        net_states: tp.Any = None,
    ) -> tp.Tuple[tp.Any, States]:
        return utils.inject_dependencies(self.pred_step)(
            net_params=net_params,
            x=x,
            net_states=net_states,
        )

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
        self.maybe_initialize(mode=Mode.pred, x=x)

        method = (
            self.pred_step_internal if self.run_eagerly else self.pred_step_internal_jit
        )

        with hooks.hooks_context():
            y_pred, state_updates = method(
                self.states.net_params,
                x,
                self.states.net_states,
            )

        self.states = self.states.merge(state_updates)

        return y_pred


class Optimizer:
    r"""A Module that wraps around `optax` optimizers."""

    def __init__(
        self,
        *optimizer: optax.GradientTransformation,
        lr_schedule: tp.Optional[
            tp.Callable[[int, tp.Optional[jnp.ndarray]], jnp.ndarray]
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


# class ModelBaseOld(Module):
#     def __init__(
#         self,
#         module: Module,
#         loss: tp.Union[tp.Callable, tp.List, tp.Dict, None] = None,
#         metrics: tp.Union[tp.Callable, tp.List, tp.Dict, None] = None,
#         optimizer: tp.Union["Optimizer", optax.GradientTransformation, None] = None,
#         run_eagerly: bool = False,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)

#         self.module = module
#         self.loss = Losses(loss) if loss is not None else None
#         self.metrics = Metrics(metrics)
#         self.optimizer = (
#             optimizer
#             if isinstance(optimizer, Optimizer)
#             else Optimizer(optimizer)
#             if optimizer is not None
#             else None
#         )
#         self._jit_functions()
#         self.initial_metrics_state: tp.Optional[tp.Dict[str, tp.Any]] = None
#         self.run_eagerly = run_eagerly

#         utils.wraps(self.module)(self)

#     def _jit_functions(self):
#         super()._jit_functions()
#         self.predict_fn_jit = elegy_jit(self.predict_fn, modules=self)
#         self.test_fn_jit = elegy_jit(self.test_fn, modules=self)
#         self.train_fn_jit = elegy_jit(self.train_fn, modules=self)

#     def __getstate__(self):
#         d = super().__getstate__()
#         del d["predict_fn_jit"]
#         del d["test_fn_jit"]
#         del d["train_fn_jit"]
#         return d

#     def call(self, *args, **kwargs):
#         return self.module(*args, **kwargs)

#     def reset_metrics(self, hard: bool = False):
#         if hard:
#             self.metrics.reset()
#             self.initial_metrics_state = None
#         elif self.initial_metrics_state is not None:
#             self.metrics.set_parameters(self.initial_metrics_state)

#     def predict_fn(self, x: tp.Any = ()):

#         x_args, x_kwargs = utils.get_input_args(x, training=module.is_training())
#         y_pred = utils.inject_dependencies(self)(*x_args, **x_kwargs)

#         return y_pred

#     def predict_step(self, x: tp.Any = ()):
#         with module.training_context(training=False):
#             return self.predict_fn(x=x)

#     def predict_step_jit(self, x: tp.Any = ()):
#         with module.training_context(training=False):
#             return self.predict_fn_jit(x)

#     def predict_on_batch(
#         self, x: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple]
#     ) -> tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple]:
#         """
#         Returns predictions for a single batch of samples.

#         Arguments:
#             x: Input data. A Numpy/Jax array (or array-like), or possibly
#                 nested python structure of dict, list, tuple that contain
#                 arrays as leafs.

#         Returns:
#             Jax array(s) of predictions.

#         Raises:
#             ValueError: In case of mismatch between given number of inputs and
#                 expectations of the model.
#         """
#         self.maybe_initialize(mode=Mode.predict, x=x)

#         method = self.predict_step if self.run_eagerly else self.predict_step_jit

#         return method(x=x)

#     def loss_fn(
#         self,
#         x: tp.Any = (),
#         y: tp.Any = None,
#         sample_weight: tp.Optional[np.ndarray] = None,
#         class_weight: tp.Optional[np.ndarray] = None,
#     ):
#         y_pred = self.predict_fn(x)

#         if self.loss is not None:
#             loss_logs = self.loss(
#                 x=x,
#                 y_true=y,
#                 y_pred=y_pred,
#                 sample_weight=sample_weight,
#                 class_weight=class_weight,
#                 training=module.is_training(),
#                 parameters=self.module.get_parameters(trainable=True),
#                 states=self.module.get_parameters(trainable=False),
#             )
#         else:
#             loss_logs = {}

#         hooks_losses_logs = module.get_losses()

#         if hooks_losses_logs is None:
#             hooks_losses_logs = {}

#         loss = sum(loss_logs.values()) + sum(hooks_losses_logs.values())

#         total_loss_logs = {}
#         total_loss_logs.update(hooks_losses_logs)
#         total_loss_logs.update(loss_logs)
#         total_loss_logs["loss"] = loss

#         return loss, y_pred, total_loss_logs

#     def test_fn(
#         self,
#         x: tp.Any = (),
#         y: tp.Any = None,
#         sample_weight: tp.Optional[np.ndarray] = None,
#         class_weight: tp.Optional[np.ndarray] = None,
#         get_gradients: bool = False,
#     ) -> tp.Tuple[np.ndarray, tp.Dict, tp.Optional[tp.Dict]]:

#         if get_gradients:
#             (loss, y_pred, total_loss_logs), grads = module.value_and_grad(
#                 self.loss_fn, modules=self.module
#             )(x, y, sample_weight, class_weight)
#         else:
#             grads = None
#             loss, y_pred, total_loss_logs = self.loss_fn(
#                 x, y, sample_weight, class_weight
#             )

#         logs = self.metrics(
#             total_loss_logs,
#             x=x,
#             y_true=y,
#             y_pred=y_pred,
#             sample_weight=sample_weight,
#             class_weight=class_weight,
#             training=module.is_training(),
#             parameters=self.module.get_parameters(trainable=True),
#             states=self.module.get_parameters(trainable=False),
#         )

#         return loss, logs, grads

#     def test_step(
#         self,
#         x: tp.Any = (),
#         y: tp.Any = None,
#         sample_weight: tp.Optional[np.ndarray] = None,
#         class_weight: tp.Optional[np.ndarray] = None,
#         get_gradients: bool = False,
#     ) -> tp.Tuple[np.ndarray, tp.Dict, tp.Optional[tp.Dict]]:

#         with module.training_context(training=False), module.hooks_context():
#             return self.test_fn(
#                 x=x,
#                 y=y,
#                 sample_weight=sample_weight,
#                 class_weight=class_weight,
#                 get_gradients=get_gradients,
#             )

#     def test_step_jit(
#         self,
#         x: tp.Any = (),
#         y: tp.Any = None,
#         sample_weight: tp.Optional[np.ndarray] = None,
#         class_weight: tp.Optional[np.ndarray] = None,
#     ):
#         with module.training_context(training=False), module.hooks_context():
#             return self.test_fn_jit(x, y, sample_weight, class_weight)

#     def test_on_batch(
#         self,
#         x: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple],
#         y: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple, None] = None,
#         sample_weight: tp.Optional[jnp.ndarray] = None,
#         class_weight: tp.Optional[jnp.ndarray] = None,
#     ) -> tp.Dict[str, jnp.ndarray]:
#         """
#         Test the model on a single batch of samples.

#         Arguments:
#             x: Input data. It could be:

#                 - A Numpy array (or array-like), or a list
#                     of arrays (in case the model has multiple inputs).
#                 - A dict mapping input names to the corresponding arrays, if
#                     the model has named inputs.
#             y: Target data. Like the input data `x`, it could be either Numpy
#                 array(s) or Jax array(s).
#             sample_weight: Optional array of the same length as x, containing
#                 weights to apply to the model's loss for each sample. In the case of
#                 temporal data, you can pass a 2D array with shape (samples,
#                 sequence_length), to apply a different weight to every timestep of
#                 every sample.

#         Returns:
#             A `logs` dictionary of containing the main `loss` as well as all
#             other losses and metrics.
#         Raises:
#             ValueError: In case of invalid user-provided arguments.
#         """
#         self.maybe_initialize(
#             mode=Mode.test,
#             x=x,
#             y=y,
#             sample_weight=sample_weight,
#             class_weight=class_weight,
#         )

#         method = self.test_step if self.run_eagerly else self.test_step_jit

#         loss, logs, grads = method(
#             x=x, y=y, sample_weight=sample_weight, class_weight=class_weight
#         )

#         return logs

#     def train_fn(
#         self,
#         x: tp.Any = (),
#         y: tp.Any = None,
#         sample_weight: tp.Optional[np.ndarray] = None,
#         class_weight: tp.Optional[np.ndarray] = None,
#     ) -> tp.Dict[str, tp.Any]:
#         assert self.optimizer is not None

#         loss, logs, grads = self.test_fn(
#             x=x,
#             y=y,
#             sample_weight=sample_weight,
#             class_weight=class_weight,
#             get_gradients=True,
#         )

#         assert grads is not None

#         lr = self.optimizer.get_effective_learning_rate()

#         if lr is not None:
#             logs["lr"] = lr

#         parameters = self.module.get_parameters(trainable=True)

#         parameters = self.optimizer(parameters, grads)

#         if module.can_update():
#             self.module.set_parameters(parameters)

#         return logs

#     def train_step(
#         self,
#         x: tp.Any = (),
#         y: tp.Any = None,
#         sample_weight: tp.Optional[np.ndarray] = None,
#         class_weight: tp.Optional[np.ndarray] = None,
#     ) -> tp.Dict[str, tp.Any]:

#         with module.training_context(training=True), module.hooks_context():
#             return self.train_fn(
#                 x=x, y=y, sample_weight=sample_weight, class_weight=class_weight
#             )

#     def train_step_jit(
#         self,
#         x: tp.Any = (),
#         y: tp.Any = None,
#         sample_weight: tp.Optional[np.ndarray] = None,
#         class_weight: tp.Optional[np.ndarray] = None,
#     ):
#         with module.training_context(training=True), module.hooks_context():
#             outputs = self.train_fn_jit(x, y, sample_weight, class_weight)

#         return outputs

#     def train_on_batch(
#         self,
#         x: tp.Union[np.ndarray, tp.Mapping[str, tp.Any], tp.Tuple],
#         y: tp.Union[np.ndarray, tp.Mapping[str, tp.Any], tp.Tuple, None] = None,
#         sample_weight: tp.Optional[np.ndarray] = None,
#         class_weight: tp.Optional[np.ndarray] = None,
#     ) -> tp.Dict[str, np.ndarray]:
#         """
#         Runs a single gradient update on a single batch of data.

#         Arguments:
#             x: Input data. It could be:

#                 - A Numpy array (or array-like), or a iterable of arrays
#                     (in case the model has multiple inputs).
#                 - A dict mapping input names to the corresponding arrays,
#                     if the model has named inputs.
#             y: Target data. Like the input data `x`, it could be either Numpy
#                 array(s) or Jax array(s). It should be consistent with `x`
#                 (you cannot have Numpy inputs and array targets, or inversely).
#             sample_weight: Optional array of the same length as x, containing
#                 weights to apply to the model's loss for each sample. In the case of
#                 temporal data, you can pass a 2D array with shape (samples,
#                 sequence_length), to apply a different weight to every timestep of
#                 every sample.
#             class_weight: Optional dictionary mapping class indices (integers) to a
#                 weight (float) to apply to the model's loss for the samples from this
#                 class during training. This can be useful to tell the model to "pay
#                 more attention" to samples from an under-represented class.

#         Returns:
#             A `logs` dictionary of containing the main `loss` as well as all
#             other losses and metrics.

#         Raises:
#             ValueError: In case of invalid user-provided arguments.
#         """
#         self.maybe_initialize(
#             mode=Mode.train,
#             x=x,
#             y=y,
#             sample_weight=sample_weight,
#             class_weight=class_weight,
#         )

#         method = self.train_step if self.run_eagerly else self.train_step_jit

#         return method(x=x, y=y, sample_weight=sample_weight, class_weight=class_weight)

#     def maybe_initialize(
#         self,
#         mode: Mode,
#         x: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple] = (),
#         y: tp.Union[jnp.ndarray, tp.Mapping[str, tp.Any], tp.Tuple, None] = None,
#         sample_weight: tp.Optional[jnp.ndarray] = None,
#         class_weight: tp.Optional[jnp.ndarray] = None,
#     ):

#         with module.init_context(can_update=False), module.training_context(
#             training=True
#         ), module.hooks_context():
#             assert self.module is not None

#             if not self.module.initialized:
#                 self.predict_fn(x=x)
#                 self.module.initialized = True

#             if mode == Mode.predict:
#                 return

#             if self.metrics is not None and not self.metrics.initialized:

#                 self.test_fn(
#                     x=x,
#                     y=y,
#                     sample_weight=sample_weight,
#                     class_weight=class_weight,
#                 )
#                 self.metrics.initialized = True

#                 self.initial_metrics_state = self.metrics.get_parameters(
#                     trainable=False
#                 )

#             if mode == Mode.test:
#                 return

#             if self.optimizer is not None and not self.optimizer.initialized:
#                 self.train_fn(
#                     x=x,
#                     y=y,
#                     sample_weight=sample_weight,
#                     class_weight=class_weight,
#                 )
#                 self.optimizer.initialized = True

#     def summary(
#         self, x, depth: int = 2, tablefmt: str = "fancy_grid", **tablulate_kwargs
#     ):
#         """
#         Prints a summary of the network.

#         Arguments:
#             x: A sample of inputs to the network.
#             depth: The level number of nested level which will be showed.
#                 Information about summaries from modules deeper than `depth`
#                 will be aggregated together.
#             tablefmt: A string represeting the style of the table generated by
#                 `tabulate`. See
#                 [python-tabulate](https://github.com/astanin/python-tabulate)
#                 for more options.
#             tablulate_kwargs: Additional keyword arguments passed to `tabulate`.
#                 See [python-tabulate](https://github.com/astanin/python-tabulate)
#                 for more options.
#         """
#         self.maybe_initialize(mode=Mode.predict, x=x)

#         with hooks_context(summaries=True):
#             self.predict_fn(x)

#             summaries = get_summaries()

#         assert summaries is not None

#         def format_output(outputs) -> str:
#             file = StringIO()
#             outputs = jax.tree_map(lambda x: f"{x.shape}{{pad}}  {x.dtype}", outputs)
#             yaml.safe_dump(
#                 outputs, file, default_flow_style=False, indent=2, explicit_end=False
#             )
#             return file.getvalue().replace("\n...", "")

#         def format_size(size):
#             return (
#                 f"{size / 1e9 :,.1f} GB"
#                 if size > 1e9
#                 else f"{size / 1e6 :,.1f} MB"
#                 if size > 1e6
#                 else f"{size / 1e3 :,.1f} KB"
#                 if size > 1e3
#                 else f"{size:,} B"
#             )

#         table: tp.List = [["Inputs", format_output(x), "0", "0"]]

#         for module, base_name, value in summaries:
#             base_name_parts = base_name.split("/")[1:]
#             module_depth = len(base_name_parts)

#             if module_depth > depth:
#                 continue

#             include_submodules = module_depth == depth

#             params_count = (
#                 module.parameters_size(include_submodules) if module is not None else 0
#             )
#             params_size = (
#                 module.parameters_bytes(include_submodules) if module is not None else 0
#             )
#             states_count = (
#                 module.states_size(include_submodules) if module is not None else 0
#             )
#             states_size = (
#                 module.states_bytes(include_submodules) if module is not None else 0
#             )
#             class_name = f"({module.__class__.__name__})" if module is not None else ""

#             base_name = "/".join(base_name_parts)

#             if not base_name:
#                 base_name = "Outputs"

#             table.append(
#                 [
#                     f"{base_name}{{pad}}  {class_name}",
#                     format_output(value),
#                     f"{params_count:,}{{pad}}    {format_size(params_size)}"
#                     if params_count > 0
#                     else "0",
#                     f"{states_count:,}{{pad}}    {format_size(states_size)}"
#                     if states_count > 0
#                     else "0",
#                 ]
#             )

#         # add papdding
#         for col in range(4):
#             max_length = max(
#                 len(line.split("{pad}")[0])
#                 for row in table
#                 for line in row[col].split("\n")
#             )

#             for row in table:
#                 row[col] = "\n".join(
#                     line.format(
#                         pad=" " * (max_length - len(line.rstrip().split("{pad}")[0]))
#                     )
#                     for line in row[col].rstrip().split("\n")
#                 )

#         print(
#             "\n"
#             + tabulate(
#                 table,
#                 headers=[
#                     "Layer",
#                     "Outputs Shape",
#                     "Trainable\nParameters",
#                     "Non-trainable\nParameters",
#                 ],
#                 tablefmt=tablefmt,
#                 **tablulate_kwargs,
#             )
#         )

#         params_count = self.parameters_size()
#         params_size = self.parameters_bytes()
#         states_count = self.states_size()
#         states_size = self.states_bytes()
#         total_count = params_count + states_count
#         total_size = params_size + states_size

#         print(
#             tabulate(
#                 [
#                     [
#                         f"Total Parameters:",
#                         f"{total_count:,}",
#                         f"{format_size(total_size)}" if total_count > 0 else "",
#                     ],
#                     [
#                         f"Trainable Parameters:",
#                         f"{params_count:,}",
#                         f"{format_size(params_size)}" if params_count > 0 else "",
#                     ],
#                     [
#                         f"Non-trainable Parameters:",
#                         f"{states_count:,}",
#                         f"{format_size(states_size)}" if states_count > 0 else "",
#                     ],
#                 ],
#                 tablefmt="plain",
#             )
#             + "\n"
#         )


# class Losses(Module):
#     def __init__(self, losses):
#         super().__init__(name="losses")
#         self.losses = losses

#     def call(self, **kwargs):

#         logs = {}

#         for context, val in self.apply_recursive((), self.losses, **kwargs):
#             loss_name = self.get_unique_loss_name(context, logs)
#             logs[loss_name] = val

#         return logs

#     def apply_recursive(self, context: tp.Tuple[str, ...], losses, **kwargs):

#         if isinstance(losses, tp.Callable):
#             name = (
#                 losses.name
#                 if isinstance(losses, Loss)
#                 else utils.lower_snake_case(losses.__name__)
#             )
#             context += (name,)
#             val = utils.inject_dependencies(losses)(**kwargs)

#             if isinstance(val, tp.Dict):
#                 for name, val in val.items():
#                     yield context + (name,), val
#             else:
#                 yield context, val

#         elif isinstance(losses, (tp.Tuple, tp.List)):
#             for loss in losses:
#                 yield from self.apply_recursive(context, loss, **kwargs)
#         elif isinstance(losses, tp.Dict):
#             for name, loss in losses.items():
#                 yield from self.apply_recursive(context + (name,), loss, **kwargs)
#         else:
#             raise TypeError(f"Invalid type {type(losses)}")

#     def get_unique_loss_name(self, context, logs):
#         context = list(context)

#         if not context[0].endswith("loss"):
#             context[0] += "_loss"

#         name = "/".join(context)

#         if name not in logs:
#             return name

#         i = 1
#         while f"{name}_{i}" in logs:
#             i += 1

#         return f"{name}_{i}"


# class LossMetrics(Metric):
#     def call(self, logs):

#         count = self.add_parameter("count", initializer=jnp.zeros, trainable=False)
#         total = self.add_parameter(
#             "total",
#             initializer=lambda *args: jax.tree_map(lambda x: jnp.array(0.0), logs),
#             trainable=False,
#         )

#         count += 1
#         total = jax.tree_multimap(lambda a, b: a + b, total, logs)

#         self.update_parameter("count", count)
#         self.update_parameter("total", total)

#         logs = jax.tree_map(lambda total: total / count, total)

#         return logs


# class Metrics(Metric):
#     def __init__(self, metrics, **kwargs):
#         super().__init__(**kwargs)
#         self.metrics = metrics if metrics is not None else tuple()

#     def call(self, logs, **kwargs):

#         # Loss logs
#         logs = LossMetrics()(logs)

#         # Metric logs
#         for context, val in self.apply_recursive((), self.metrics, **kwargs):
#             name = "/".join(context)
#             name = self.get_unique_metric_name(logs, name)
#             logs[name] = val

#         return logs

#     def apply_recursive(self, context: tp.Tuple[str, ...], metrics, **kwargs):

#         if isinstance(metrics, tp.Callable):

#             name = (
#                 metrics.name
#                 if isinstance(metrics, module.Module)
#                 else utils.lower_snake_case(metrics.__name__)
#             )
#             context += (name,)
#             value = utils.inject_dependencies(metrics)(**kwargs)

#             if isinstance(value, tp.Dict):
#                 for name, value in value.items():
#                     yield context + (name,), value
#             else:
#                 yield context, value

#         elif isinstance(metrics, (tp.Tuple, tp.List)):
#             for loss in metrics:
#                 yield from self.apply_recursive(context, loss, **kwargs)
#         elif isinstance(metrics, tp.Dict):
#             for name, loss in metrics.items():
#                 yield from self.apply_recursive(context + (name,), loss, **kwargs)
#         else:
#             raise TypeError(f"Invalid type {type(metrics)}")

#     def get_unique_metric_name(self, logs, name):

#         if name not in logs:
#             return name

#         i = 1
#         while f"{name}_{i}" in logs:
#             i += 1

#         return f"{name}_{i}"
