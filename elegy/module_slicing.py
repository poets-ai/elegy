import networkx as nx
import elegy
from elegy.module import Module
import jax
import itertools
import typing as tp
import numpy as np

import sys
from . import module

module.module_slicing = sys.modules[__name__]


def slice_module_from_to(
    module: Module,
    start_module: tp.Union[Module, str, None],
    end_module: tp.Union[Module, str, None, tp.List[tp.Union[Module, str, None]]],
    sample_input: np.ndarray,
) -> Module:
    assert not isinstance(
        start_module, (tp.Tuple, tp.List)
    ), "Multiple inputs not yet supported"

    # get info about the module structure via summaries
    model = elegy.Model(module)
    with elegy.hooks_context(summaries=True):
        model.predict_fn(sample_input)
        summaries = elegy.get_summaries()

    edges = [Edge(summ) for summ in summaries]
    if start_module in ["/input", "input"]:
        start_module = None
    start_id = get_input_id(edges, start_module)
    if not isinstance(end_module, (tp.Tuple, tp.List)):
        end_module = [end_module]
    end_ids = [
        get_output_id(edges, m)
        if m not in ["/input", "input"]
        else get_input_id(edges, None)
        for m in end_module
    ]

    graph = construct_graph(edges)
    dag_paths = [find_dag_path(graph, start_id, end_id) for end_id in end_ids]
    tree = combine_paths(dag_paths)  # not really a tree
    submodule = SlicedModule(tree)
    return submodule


class Edge:
    """A struct to hold edge data"""

    def __init__(self, summary: tp.Tuple[Module, str, np.ndarray, tp.Any]):
        self.module = summary[0]
        # remove the full module name, leave the leading '/'
        self.modulename = (
            summary[1][summary[1].find("/") :] if "/" in summary[1] else "/"
        )
        # convert the output and input arrays in the summary to unique IDs as returned by id()
        self.output_ids = jax.tree_leaves(jax.tree_map(id, summary[2]))
        self.input_ids = jax.tree_map(id, summary[3])


def search_edges(
    edges: tp.List[Edge], searchtarget: tp.Union[Module, str, None]
) -> Edge:
    """Searches 'edges' for 'searchtarget' which can be a module, name of a module or None"""
    if searchtarget is None:
        # None means input/output of the full module, which is the last edge
        return edges[-1]
    elif isinstance(searchtarget, str):
        # search by name, with or without leading '/'
        if not searchtarget.startswith("/"):
            searchtarget = "/" + searchtarget
        edges = [e for e in edges if e.modulename == searchtarget]
    elif isinstance(searchtarget, Module):
        # search by reference
        edges = [e for e in edges if e.module == searchtarget]
    assert len(edges) > 0, f"Could not find module {searchtarget}"
    assert len(edges) < 2, f"Found {len(edges)} modules for {searchtarget}"
    return edges[0]


def get_input_id(edges: tp.List[Edge], module: tp.Union[Module, str, None]) -> int:
    """Searches for module in the list of edges and returns the ID of its input array"""
    edge = search_edges(edges, module)
    input_ids = jax.tree_leaves(edge.input_ids)
    assert len(input_ids) == 1, "Multi-input modules not yet supported"
    return input_ids[0]


def get_output_id(edges: tp.List[Edge], module: tp.Union[Module, str, None]) -> int:
    """Searches for module in the list of edges and returns the ID of its output array"""
    edge = search_edges(edges, module)
    assert len(edge.output_ids) == 1, "Multi-output modules not yet supported"
    return edge.output_ids[0]


def merge_args_kwargs(*args, **kwargs) -> tp.List[tp.Tuple[tp.Any, tp.Any]]:
    """Merges args and kwargs and their indices to a list of tuples
    e.g. merge_args_kwargs(0, 77, a=-2) returns [(0,0), (1,77), ('a',-2)]"""
    return list(enumerate(args)) + list(kwargs.items())


def split_merged_args_kwargs(
    args_kwargs: tp.List[tp.Tuple[tp.Any, tp.Any]]
) -> tp.Tuple[tp.Tuple, tp.Dict]:
    """Reverse operation of merge_args_kwargs().
    e.g. split_merged_args_kwargs([(0,0), (1,77), ('a':-2)]) -> (0,77), {'a':-2}"""
    args, kwargs = list(), dict()
    for key, value in args_kwargs:
        if isinstance(key, int):
            args.append(value)
        else:
            kwargs[key] = value
    return tuple(args), kwargs


def construct_graph(edges: tp.List[Edge]) -> nx.DiGraph:
    """Constructs a directed graph with IDs of input/output arrays representing the nodes
    and modules (and some more infos) representing the edges"""
    G = nx.DiGraph()
    for e in edges:
        merged_args_kwargs = merge_args_kwargs(*e.input_ids[0], **e.input_ids[1])
        inout_combos = itertools.product(merged_args_kwargs, enumerate(e.output_ids))
        for ((inkey, input_id), (outkey, output_id)) in inout_combos:
            depth = e.modulename.count("/")
            # it can happen that there are multiple connections between two nodes
            # e.g. when a simple parent module has only one child module
            # use the one with the lowest depth, i.e. the parent module
            if ((input_id, output_id) not in G.edges) or (
                G[input_id][output_id].depth > depth
            ):
                G.add_edge(
                    input_id,
                    output_id,
                    inkey=inkey,
                    outkey=outkey,
                    depth=depth,
                    **e.__dict__,
                )

    # adding dummy edges from inputs to inputs
    e = edges[-1]  # edge representing the full module
    merged_args_kwargs = merge_args_kwargs(*e.input_ids[0], **e.input_ids[1])
    for key, node_id in merged_args_kwargs:
        G.add_edge(
            node_id,
            node_id,
            inkey=key,
            outkey=key,
            depth=0,
            module=lambda x: x,
            modulename="Inputs",
            input_ids=[node_id],
            output_ids=[node_id],
        )
    return G


def are_paths_computationally_equivalent(path0: nx.DiGraph, path1: nx.DiGraph) -> bool:
    """Checks two paths for computaional equivalence i.e. whether or not they differ only in depth of modules.
    E.g if node B is computed by a module composed of several submodules with subnodes B0 and B1
    then paths A->B->C and A->B0->B1->B->C are computationally equivalent.
    On the other hand, this does not apply to branches A->B->C vs A->D->C.
    Importantly, the edge["inkey"] attributes must be the same:
    A->C != A->B->C if C if computed by a dual-input module (e.g. C = A+B)"""
    # traverse both paths and check if nodes path0 are in path1 or vice versa
    # get nodes from both paths, make sure they are ordered
    # skip the first one assuming both have the same source node
    nodes0 = list(nx.dfs_postorder_nodes(path0))[::-1][1:]
    nodes1 = list(nx.dfs_postorder_nodes(path1))[::-1][1:]
    while len(nodes0) and len(nodes1):
        # currently traversed nodes from both paths
        n0, n1 = nodes0[0], nodes1[0]

        if n0 in nodes1:
            # current node of path0 is in path1, still need to check 'inkey'
            inkey0 = path0.get_edge_data(*list(path0.in_edges(n0))[0])["inkey"]
            inkey1 = path1.get_edge_data(*list(path1.in_edges(n0))[0])["inkey"]
            if inkey0 == inkey1:
                # all ok, continue traversing paths
                nodes1 = nodes1[nodes1.index(n0) + 1 :]
                nodes0 = nodes0[1:]
                continue
            else:
                # inkey is not the same, must be a multi-input module -> reject
                return False
        elif n1 in nodes0:
            # current node of path1 is in path0, still need to check 'inkey'
            inkey0 = path0.get_edge_data(*list(path0.in_edges(n1))[0])["inkey"]
            inkey1 = path1.get_edge_data(*list(path1.in_edges(n1))[0])["inkey"]
            if inkey0 == inkey1:
                # all ok, continue traversing paths
                nodes0 = nodes0[nodes0.index(n1) + 1 :]
                nodes1 = nodes1[1:]
                continue
            else:
                # inkey is not the same, must be a multi-input module -> reject
                return False
        else:
            # neither path contains the current node of the other path -> reject
            return False
    if len(nodes0) > 0 or len(nodes1) > 0:
        # should not happen because our paths have the same first and last nodes
        return False
    # traversed both paths until the end
    return True


def filter_computationally_equivalent_paths(
    paths: tp.List[nx.DiGraph],
) -> tp.List[nx.DiGraph]:
    """Removes paths with deep modules if there are paths with equivalent, shallow modules.
    E.g: remove A->B0->B1->B->C in favor of A->B->C"""
    filtered = set()  # contains indices of paths to be removed
    for i, j in itertools.combinations(range(len(paths)), 2):
        if i in filtered or j in filtered:
            continue
        if are_paths_computationally_equivalent(paths[i], paths[j]):
            # keep the shorter path
            if len(paths[i]) > len(paths[j]):
                filtered.add(i)
            else:
                filtered.add(j)
    paths = [paths[i] for i in range(len(paths)) if i not in filtered]
    return paths


def find_dag_path(graph: nx.DiGraph, start_node: int, end_node: int) -> nx.DiGraph:
    """Returns a new (possibly multi-path) graph with only nodes and edges from start_node to end_node"""
    startname = list(graph[start_node].values())[0]["modulename"]
    endname = list(graph.reverse()[end_node].values())[0]["modulename"]

    try:
        edge_paths = list(
            nx.all_simple_edge_paths(graph, start_node, end_node)
        )  # list of lists of tuples
        if len(edge_paths) == 0:
            if start_node == end_node and (start_node, end_node) in graph.edges:
                # input -> input
                edge_paths = [[(start_node, end_node)]]
            else:
                raise nx.NetworkXNoPath
    except nx.NetworkXNoPath:
        raise RuntimeError(
            f"No path from {startname} to {endname}. Make sure all operations inbetween are performed by modules."
        ) from None

    graph_paths = [
        nx.edge_subgraph(graph, path) for path in edge_paths
    ]  # list of nx.DiGraphs
    graph_paths = filter_computationally_equivalent_paths(graph_paths)
    dag_graph = nx.algorithms.compose_all(graph_paths)
    # dag_graph is unordered, need to mark input and output edges
    for _, _, edgedata in dag_graph.out_edges(start_node, data=True):
        edgedata["is_input"] = True
    for _, _, edgedata in dag_graph.in_edges(end_node, data=True):
        edgedata["is_output"] = True
    return dag_graph


def combine_paths(paths: tp.List[nx.DiGraph]) -> nx.DiGraph:
    return nx.algorithms.compose_all(paths)


class SlicedModule(elegy.Module):
    def __init__(self, tree: nx.DiGraph):
        super().__init__()
        # adding the all modules as attributes so that they get recognized by .get_parameters()
        for edge in tree.edges.values():
            attrname = edge["modulename"][1:].replace("/", "_")
            setattr(self, attrname, edge["module"])

        assert not hasattr(
            self, "_tree"
        ), 'Modules with the name "_tree" are prohibited'  # can this happen?
        self._tree = tree

    def call(self, x: tp.Any) -> tp.Union[tp.Any, tp.Tuple[tp.Any]]:
        input_nodes = [
            nodes[0]
            for nodes, edge in self._tree.edges.items()
            if edge.get("is_input", False)
        ]

        # should not happen
        assert len(set(input_nodes)) > 0, "could not find any input nodes"
        assert len(set(input_nodes)) < 2, "multi-inputs not yet supported"
        start_node = input_nodes[0]

        outputs = self.visit_node(start_node, x, deferred_call_args=dict())

        outputs = tuple(outputs)
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs

    def visit_edge(
        self, edge: tp.Dict, x: tp.Any, deferred_call_args: tp.Dict
    ) -> tp.Any:
        """Performs the operation to get from node A to node B which the parameter "edge" connects"""
        n_inputs = len(jax.tree_leaves(edge["input_ids"]))
        if n_inputs == 1:
            # a single-input module, simply call it with the input
            x = edge["module"](x)
        else:
            # multi-input module
            # check if all the inputs are ready
            call_args = deferred_call_args.get(edge["modulename"], dict())
            call_args[edge["inkey"]] = x
            if len(call_args) == n_inputs:
                # all inputs are ready, call module
                args, kwargs = split_merged_args_kwargs(call_args.items())
                x = edge["module"](*args, **kwargs)
                del deferred_call_args[edge["modulename"]]
            else:
                # still missing some inputs, continue traversing the graph
                deferred_call_args[edge["modulename"]] = call_args
                return DeferredCall

        if isinstance(x, (tuple, list)):
            # XXX: what if the whole tuple/list is needed as input later?
            x = x[edge["outkey"]]

        return x

    def visit_node(
        self, node: int, x: tp.Any, deferred_call_args: tp.Dict
    ) -> tp.List[tp.Any]:
        """Recursively visits all nodes starting from the parameter "node" and collects outputs."""
        outputs = []
        for nextnode, edge in self._tree[node].items():
            y = self.visit_edge(edge, x, deferred_call_args)
            if y == DeferredCall:
                # visited edge module is missing some inputs, will come back here later
                continue
            if edge.get("is_output", False):
                outputs.append(y)
            if node != nextnode:
                outputs.extend(self.visit_node(nextnode, y, deferred_call_args))
            # else: input -> input

        return outputs


class DeferredCall:
    """Dummy class that indicates that a call has to be deferred"""

    ...
