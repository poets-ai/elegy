import typing as tp

import numpy as np
import jax
import elegy


#TODO: docs




def slice_model(
    model: elegy.Model,
    start_module: tp.Union[elegy.Module, str, None],
    end_module: tp.Union[elegy.Module, str, None, tp.List[tp.Union[elegy.Module, str, None]]],
    sample_input: np.ndarray,
) -> elegy.Model:
    
    model.maybe_initialize(elegy.types.Mode.pred, x=sample_input)

    with elegy.hooks.context(named_call=True), jax.disable_jit():
        jaxpr = jax.make_jaxpr(model.pred_step, static_argnums=[2,3] )(sample_input, model.states, False, False)
    
    n_inputs            = len(jax.tree_leaves(sample_input))
    toplevel_input_vars = jaxpr.jaxpr.invars[:n_inputs]

    state_vars          = jaxpr.jaxpr.invars[n_inputs:]

    if not isinstance(end_module, (list, tuple)):
        end_module = [end_module]
    full_eq_path = []
    all_output_vars = []
    all_input_vars = []
    for end in end_module:
        eq_path, input_vars, output_vars, unresolved_vars, env = analyze_jaxpr(jaxpr.jaxpr, start_module, end)
    
        #if any([ iv in toplevel_input_vars for iv in input_vars ]):       #TODO: incorrect
        #    raise RuntimeError('Unresolved input')

        stateless_input_vars = [iv for iv in input_vars if not iv.intersection(state_vars)]
        full_eq_path = combine_eq_paths(full_eq_path, eq_path)
        all_output_vars += output_vars
        all_input_vars = list_union(all_input_vars, stateless_input_vars)
    return SlicedModule(model.module, full_eq_path, all_input_vars, all_output_vars)

def combine_eq_paths(eq_path0, eq_path1):
    new_path = []
    for eq in eq_path0:
        if eq not in eq_path1:
            new_path.append(eq)
        else:
            ix = eq_path1.index(eq)
            new_path += eq_path1[:ix]
            eq_path1  = eq_path1[ix+1:]
            new_path.append(eq)
    new_path += eq_path1
    return new_path


class Environment(dict):
    def __getitem__(self, var):
        if isinstance(var, jax.core.Literal):
            #literals are for some reason not hashable
            return var.val
        else:
            return super().__getitem__(var)
    
    def __setitem__(self, var, value):
        if isinstance(var, jax.core.Literal):
            #ignore, literals are constant
            pass
        else:
            super().__setitem__(var, value)
    
    def __contains__(self, var):
        if isinstance(var, jax.core.Literal):
            return True
        else:
            return super().__contains__(var)



class EquivalentVars(set):
    '''A set of jax.core.Var that are equivalent among different levels of jaxprs'''
    pass


def path_to_str(path: tp.Tuple[str]) -> str:
    return '/'+'/'.join(path)

def normalize_module_path(module_path: str) -> str:
    if module_path is None:
        return '/'
    return module_path if module_path.startswith('/') else '/'+module_path

def list_intersection(list0: tp.List[tp.Any], list1: tp.List[tp.Any]):
    return [x for x in list0 if x in list1]

def list_difference(list0: tp.List[tp.Any], list1: tp.List[tp.Any]):
    return [x for x in list0 if x not in list1]

def list_union(list0: tp.List[tp.Any], list1: tp.List[tp.Any]):
    return list0 + list_difference(list1, list0)

def update_env(env: tp.Dict[jax.core.Var, EquivalentVars], eq: jax.core.JaxprEqn):   #TODO: make immutable
    for v in eq.invars + eq.outvars:
        if v not in env:
            env[v] = EquivalentVars([v])

def analyze_jaxpr(jaxpr: jax.core.Jaxpr, start_path, end_path):
    start_path = normalize_module_path(start_path)
    end_path   = normalize_module_path(end_path)

    env = Environment( [(v, EquivalentVars([v])) for v in jaxpr.invars] )

    #search for end module
    for i, eq in enumerate(jaxpr.eqns):
        if eq.primitive.name == 'named_call':
            modulepath = path_to_str(eq.params['name'])
            if end_path.startswith(modulepath):
                if end_path == modulepath:
                    #list instead of set because jax literals are not hashable
                    unresolved_vars = list(eq.invars)
                    input_vars      = []
                    output_vars     = eq.outvars
                    eq_path         = [eq]
                    update_env(env, eq)
                else:
                    inner_jaxpr = eq.params['call_jaxpr']
                    eq_path, input_vars, output_vars, unresolved_vars, inner_env = analyze_jaxpr( inner_jaxpr, start_path, end_path )
                    env.update(inner_env)
                    for outer_v, inner_v in zip(eq.invars, inner_jaxpr.invars):
                        env[outer_v].add(inner_v)
                        inner_env[inner_v].add(outer_v)
                    unresolved_vars = [env[v] for v in unresolved_vars]
                    #input_vars = [env[v] for v in input_vars]
                break
    else:
        #end_path not found
        raise RuntimeError(f'End module {end_path} not found')
    

    #search from end to start, collecting only required equations
    if len(input_vars)==0:
        for eq in reversed(jaxpr.eqns[:i+1]):
            common_vars = list_intersection(unresolved_vars, eq.outvars)
            if len(common_vars):
                unresolved_vars = list_difference(unresolved_vars, common_vars)
                unresolved_vars = list_union(unresolved_vars, eq.invars)
                eq_path = [eq] + eq_path
                update_env(env, eq)

            if eq.primitive.name == 'named_call':
                modulepath = path_to_str(eq.params['name'])
                if start_path.startswith(modulepath):
                    if start_path == modulepath:
                        unresolved_vars = list_difference(unresolved_vars, eq.invars)
                        update_env(env, eq)
                        input_vars      = [env[v] for v in eq.invars]
                        break
                    else:
                        assert 0, NotImplemented
                        #analyze_jaxpr( eq.params['call_jaxpr'], start_path, end_path )   #not quite
                        #break
        #else:
            #start path not found, can happen in nested calls
    
    return eq_path, input_vars, output_vars, unresolved_vars, env



def get_module(parentmodule:elegy.Module, name:tp.Tuple[str]):
    if name==():
        return parentmodule
    else:
        for n in name:
            parentmodule = getattr(parentmodule, n)
        return parentmodule


class SlicedModule(elegy.Module):
    def __init__(self, mainmodule:elegy.Module, equations:tp.List[jax.core.JaxprEqn], input_vars:tp.List[jax.core.Var], output_vars):
        super().__init__()
        self.modules = dict([ (eq.params['name'], get_module(mainmodule, eq.params['name'])) for eq in equations if eq.primitive.name=='named_call'])
        self.equations = equations
        self.input_vars = input_vars
        self.output_vars = output_vars
    
    def call(self, *args):
        if len(args)!=len(self.input_vars):
            raise TypeError(f'Expected {len(self.input_vars)} inputs, received {len(args)}')

        environment: tp.Dict[jax.core.Var, tp.Any] = Environment()
        for equivar, arg in zip(self.input_vars, args):
            for v in equivar:
                environment[v] = arg
        
        for eq in self.equations:
            eq_inputs = [environment[v] for v in eq.invars if v in environment]
            if eq.primitive.name == 'named_call':
                module = self.modules[eq.params['name']]
                outputs = module(*eq_inputs)
            else:
                outputs = eq.primitive.bind(*eq_inputs, **eq.params)
            
            if isinstance(outputs, list):
                outputs = tuple(outputs)
            elif not isinstance(outputs, tuple):
                outputs = (outputs,)

            for o,v in zip(outputs, eq.outvars):
                environment[v] = o
        
        outputs = tuple(environment[v] for v in self.output_vars)
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs

