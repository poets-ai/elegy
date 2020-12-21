import optax
from optax import *
import functools
from collections import namedtuple


# extend optax.GradientTransformation with learning rate and step_functions
GradientTransformation = namedtuple(
    "GradientTransformation", optax.GradientTransformation._fields + ("lr", "step_fns")
)


# search for all optax functions that accept a `learning_rate` parameter
# replace them to return the extended GradientTransformation
for funcname in dir(optax):
    func = getattr(optax, funcname)
    if "learning_rate" in getattr(func, "__annotations__", dict()):

        def wrap_func(func):
            @functools.wraps(func)
            def newfunc(*args, **kwargs):
                lr = kwargs.get("learning_rate", args[0])
                return GradientTransformation(
                    *func(*args, **kwargs), lr=lr, step_fns=[]
                )

            return newfunc

        globals()[funcname] = wrap_func(func)

# replace optax.scale_by_schedule to return the extended GradientTransformation
@functools.wraps(optax.scale_by_schedule)
def scale_by_schedule(step_fn):
    return GradientTransformation(
        *optax.scale_by_schedule(step_fn), lr=None, step_fns=[step_fn]
    )


# replace optax.chain
@functools.wraps(optax.chain)
def chain(*gradient_transforms):
    lrates, step_fns = [], []
    for gtransform in gradient_transforms:
        if getattr(gtransform, "lr", None) is not None:
            lrates.append(gtransform.lr)
        elif getattr(gtransform, "step_fns", None) is not None:
            step_fns.extend(gtransform.step_fns)

    assert (
        len(lrates) <= 1
    ), "Monitoring multiple learning rates not supported. Use optax instead of elegy.optax"

    gt = optax.chain(*[gt[:2] for gt in gradient_transforms])
    if len(lrates) == 1:
        gt = GradientTransformation(*gt, lr=lrates[0], step_fns=step_fns)
    return gt


def find_schedule_states(state):
    """Recursively searches the (possibly nested) state for ScaleByScheduleState"""
    result = []
    if isinstance(state, optax.ScaleByScheduleState):
        result.append(state)
    elif isinstance(state, list):
        for s in state:
            result.extend(find_schedule_states(s))
    return result
