import typing as tp
import functools

import numpy as np
import jax, jax.numpy as jnp
import elegy
from elegy.module import Module
import elegy.hooks as hooks
import elegy.types as types


def slice_model(
    model: "elegy.Model",
    start: tp.Union[str, None],
    end: tp.Union[str, None, tp.List[tp.Union[str, None]]],
    sample_input: tp.Any,
) -> Module:

    if not model.initialized:
        model.predict(sample_input, initialize=True)
    model.update_modules()

    # create a jaxpr, marking all modules with named_call
    with hooks.context(named_call=True), jax.disable_jit():
        jaxpr = jax.make_jaxpr(model.pred_step, static_argnums=[2, 3])(
            sample_input, model.states, False, False
        )
    jaxpr = jaxpr.jaxpr
    jaxpr = replace_named_call_vars(jaxpr)

    n_inputs = len(jax.tree_leaves(sample_input))
    # actual module inputs
    toplevel_input_vars = jaxpr.invars[:n_inputs]
    # weights and biases
    state_vars = jaxpr.invars[n_inputs:]

    ends = end if isinstance(end, (list, tuple)) else [end]
    input_vars, output_vars, unresolved_vars, eqn_sequence = analyze_jaxpr(
        jaxpr, start, ends
    )

    stateless_unresolved_vars = [
        v for v in unresolved_vars if v not in state_vars + input_vars
    ]
    stateless_unresolved_vars = filter_literals(stateless_unresolved_vars)
    stateless_input_vars = [v for v in input_vars if v not in state_vars]
    # flatten
    output_vars = [var for vars in output_vars for var in vars]
    output_vars = [v for v in output_vars if v not in state_vars]

    if len(stateless_unresolved_vars) or None in output_vars:
        raise RuntimeError(f"No path from {start} to {end}")

    return SlicedModule(model.module, eqn_sequence, stateless_input_vars, output_vars)


def filter_literals(
    vars: tp.List[tp.Union[jax.core.Var, jax.core.Literal]]
) -> tp.List[jax.core.Var]:
    """Removes jax.core.Literal values from the input list"""
    return [v for v in vars if not isinstance(v, jax.core.Literal)]


def replace_named_call_vars(
    jaxpr: jax.core.Jaxpr,
    env: tp.Dict[jax.core.Var, jax.core.Var] = None,
    level: str = "",
) -> jax.core.Jaxpr:
    """Replaces vars of inner jaxprs with vars from outer jaxprs if they are equivalent."""
    env = env or dict()
    for inv in jaxpr.invars:
        if inv not in env:
            env[inv] = inv

    top_eqns = []
    invars = [
        env[v] if not isinstance(v, jax.core.Literal) else v for v in jaxpr.invars
    ]

    for eq in jaxpr.eqns:
        eq_outvars = [env[v] if v in env else v for v in eq.outvars]
        for outv in eq_outvars:
            env[outv] = jax.core.Var(outv.count, f"_{level}", outv.aval)

        inner_invars = [
            env[v] if not isinstance(v, jax.core.Literal) else v for v in eq.invars
        ]
        inner_outvars = [env[v] for v in eq.outvars]
        eq_params = eq.params

        if eq.primitive.name == "named_call":
            inner_jaxpr = eq.params["call_jaxpr"]
            inner_env = dict(
                (
                    [
                        (inner_v, env[outer_v])
                        for inner_v, outer_v in zip(inner_jaxpr.invars, eq.invars)
                    ]
                    + [
                        (inner_v, env[outer_v])
                        for inner_v, outer_v in zip(inner_jaxpr.outvars, eq.outvars)
                    ]
                )
            )
            inner_jaxpr = replace_named_call_vars(
                inner_jaxpr, inner_env, eq.params["name"]
            )
            eq_params = {"call_jaxpr": inner_jaxpr, "name": eq.params["name"]}

        new_eqn = jax.core.JaxprEqn(
            invars=inner_invars,
            outvars=inner_outvars,
            primitive=eq.primitive,
            params=eq_params,
            source_info=eq.source_info,
        )
        top_eqns += [new_eqn]

    outvars = [env[v] for v in jaxpr.outvars]
    new_jaxpr = jax.core.Jaxpr(
        constvars=jaxpr.constvars, invars=invars, outvars=outvars, eqns=top_eqns
    )
    return new_jaxpr


def strict_startswith(s0: str, s1: str) -> bool:
    return s0.startswith(s1) and not s0 == s1


def path_to_str(path: tp.Tuple[str]) -> str:
    """Converts a module path to a string e.g. ('A','B') -> "/A/B/" """
    pathstr = "/" + "/".join(path)
    if len(path) > 0:
        pathstr += "/"
    return pathstr


# constants, variations of the string "input"
INPUTS_STRINGS = ["input", "inputs"]
INPUTS_STRINGS += ["/" + s for s in INPUTS_STRINGS]
INPUTS_STRINGS += [s.upper() for s in INPUTS_STRINGS]


def normalize_module_path(module_path: tp.Union[str, None]) -> str:
    if module_path in [None] + INPUTS_STRINGS:
        return "/"
    if not module_path.startswith("/"):
        module_path = "/" + module_path
    if not module_path.endswith("/"):
        module_path = module_path + "/"
    return module_path


def analyze_jaxpr(
    jaxpr: jax.core.Jaxpr,
    start_path: tp.Union[None, str],
    end_paths: tp.List[tp.Union[None, str]],
    unresolved: tp.Set[jax.core.Var] = None,
):
    """Analyze a jaxpr, collecting only the necessary equations from start_path to end_path"""
    start_path = normalize_module_path(start_path)
    normed_end_paths = [normalize_module_path(e) for e in end_paths]

    output_vars = [[None]] * len(end_paths)
    input_vars = []
    unresolved = unresolved or set()
    eqn_sequence = []

    # iterate over equations in reverse order to make sure only necessary equations are collected
    for eq in reversed(jaxpr.eqns):
        if eq.primitive.name == "named_call":
            # encountered an elegy.Module

            eq_module_name = path_to_str(eq.params["name"])
            # check if it's one of the targets in end_paths
            for idx, end_path in enumerate(normed_end_paths):
                if eq_module_name == end_path:
                    # target found
                    if end_paths[idx] in INPUTS_STRINGS:
                        # special case "/input" or similar
                        # collect the input of the whole module
                        output_vars[idx] = eq.invars
                    else:
                        # otherwise collect the output of the module
                        output_vars[idx] = eq.outvars
                        unresolved = unresolved.union(eq.outvars)

            if any(
                [
                    strict_startswith(p, eq_module_name)
                    for p in normed_end_paths + [start_path]
                ]
            ):
                # module is not a target in end_paths but is a parent module
                # e.g. "/module_A" when "/module_A/module_B" is in end_paths
                # go inside the module: analyze the inner jaxpr
                inner_jaxpr = eq.params["call_jaxpr"]
                (
                    inner_invars,
                    inner_outvars,
                    inner_unresolved,
                    inner_eqns,
                ) = analyze_jaxpr(inner_jaxpr, start_path, end_paths, unresolved)
                input_vars += inner_invars
                for idx, outvar in enumerate(inner_outvars):
                    if outvar != [None]:
                        output_vars[idx] = outvar
                unresolved = inner_unresolved
                eqn_sequence = inner_eqns + eqn_sequence

        # check whether the equation is necessary
        # i.e. if some of its outputs are required by previously collected equations
        common_vars = unresolved.intersection(eq.outvars)
        if len(common_vars):
            # yes it is required
            unresolved = unresolved.difference(eq.outvars)
            unresolved = unresolved.union(filter_literals(eq.invars))
            eqn_sequence = [eq] + eqn_sequence

        # check if this equation is the specified start of slicing
        if eq.primitive.name == "named_call" and eq_module_name == start_path:
            input_vars = eq.invars
            unresolved = unresolved.difference(filter_literals(input_vars))

    return input_vars, output_vars, unresolved, eqn_sequence


class Environment(dict):
    """A dict that can deal with jax.core.Literal which is not hashable"""

    def __getitem__(self, var):
        if isinstance(var, jax.core.Literal):
            return var.val
        else:
            return super().__getitem__(var)

    def __setitem__(self, var, value):
        if isinstance(var, jax.core.Literal):
            pass
        else:
            super().__setitem__(var, value)

    def __contains__(self, var):
        if isinstance(var, jax.core.Literal):
            return True
        else:
            return super().__contains__(var)


def get_module(parentmodule: Module, path: tp.Tuple[str]):
    """Returns a submodule from the parentmodule according to path"""
    if path == ():
        return parentmodule
    else:
        for n in path:
            parentmodule = getattr(parentmodule, n)
        return parentmodule


class SlicedModule(Module):
    def __init__(
        self,
        mainmodule: Module,
        equations: tp.List[jax.core.JaxprEqn],
        input_vars: tp.List[jax.core.Var],
        output_vars: tp.List[jax.core.Var],
    ):
        super().__init__()
        for eq in equations:
            if eq.primitive.name == "named_call":
                setattr(
                    self,
                    "_".join(eq.params["name"]),
                    get_module(mainmodule, eq.params["name"]),
                )
        self.equations = equations
        self.input_vars = input_vars
        self.output_vars = output_vars

    def call(self, *args):
        if len(args) != len(self.input_vars):
            raise TypeError(
                f"Expected {len(self.input_vars)} inputs, received {len(args)}"
            )

        environment: tp.Dict[jax.core.Var, tp.Any] = Environment()
        for var, arg in zip(self.input_vars, args):
            environment[var] = arg

        # execute all equations one by one
        for eq in self.equations:
            eq_inputs = [environment[v] for v in eq.invars if v in environment]

            # if the equation is a module, execute the module instead
            if eq.primitive.name == "named_call":
                # yes it is a module
                module = getattr(self, "_".join(eq.params["name"]))
                outputs = module(*eq_inputs)
            else:
                # not a module, simple execute the primitive
                if isinstance(eq.primitive.impl, functools.partial):
                    outputs = eq.primitive.bind(*eq_inputs, **eq.params)
                else:
                    outputs = eq.primitive.impl(*eq_inputs, **eq.params)

            if isinstance(outputs, list):
                outputs = tuple(outputs)
            elif not isinstance(outputs, tuple):
                outputs = (outputs,)

            for o, v in zip(outputs, eq.outvars):
                environment[v] = o

        outputs = tuple(environment[v] for v in self.output_vars)
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs
