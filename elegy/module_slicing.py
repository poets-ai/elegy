import networkx as nx
import elegy
from elegy import Module
import jax
import itertools
import typing as tp
import numpy as np

__all__ = ["slice_module_from_to"]


def slice_module_from_to(
    module: Module,
    start_module: tp.Union[Module, str, None],
    end_module: tp.Union[Module, str, None, tp.List[tp.Union[Module, str, None]]],
    sample_input: np.ndarray,
) -> Module:
    """Creates a new submodule starting from the input of 'start_module' to the outputs of 'end_module'.
    Current limitations:
      - only one input module is supported
      - all operations between start_module and end_module must be performed by modules
        i.e. jax.nn.relu() or x+1 is not allowed but can be converted by wrapping with elegy.to_module()
      - all modules between start_module and end_module must have a single input and a single output
      - resulting module is currently not trainable
    """
    assert not isinstance(
        start_module, (tp.Tuple, tp.List)
    ), "Multiple inputs not yet supported"

    # get info about the module structure via summaries
    model = elegy.Model(module)
    with elegy.hooks_context(summaries=True):
        model.predict_fn(sample_input)
        summaries = elegy.get_summaries()

    edges = [Edge(summ) for summ in summaries]
    start_id = get_input_id(edges, start_module)
    if not isinstance(end_module, (tp.Tuple, tp.List)):
        end_module = [end_module]
    end_ids = [get_output_id(edges, m) for m in end_module]

    graph = construct_graph(edges)
    paths = [find_path(graph, start_id, end_id) for end_id in end_ids]
    tree = combine_paths(paths)
    submodule_call = construct_call(tree)
    submodule = elegy.to_module(submodule_call)()
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
    return G


def find_path(graph: nx.DiGraph, start_node: int, end_node: int) -> nx.DiGraph:
    """Returns a new graph with only nodes and edges from start_node to end_node"""
    # TODO: catch exceptions
    pathnodes = nx.shortest_path(graph, start_node, end_node)
    pathgraph = graph.subgraph(pathnodes).copy()
    # pathgraph is unordered, need to mark input and output edges
    pathgraph[pathnodes[0]][pathnodes[1]]["is_input"] = True
    pathgraph[pathnodes[-2]][pathnodes[-1]]["is_output"] = True
    return pathgraph


def combine_paths(paths: tp.List[nx.DiGraph]) -> nx.DiGraph:
    return nx.algorithms.compose_all(paths)


def construct_call(tree: nx.DiGraph) -> tp.Callable:
    """Returns a new function that represents the __call__ of the new sliced submodule"""

    def visit_edge(edge, x, next_node):
        assert edge["inkey"] == 0, "inputs other than 0 not yet implemented"
        x = edge["module"](x)

        if isinstance(x, (tuple, list)):
            # XXX: what if the whole tuple/list is needed as input later?
            x = x[edge["outkey"]]

        outputs = []
        if edge.get("is_output", False):
            outputs.append(x)

        if len(tree[next_node]):
            # continue traversing the graph if there are more edges
            for next_node, next_edge in tree[next_node].items():
                nextx = visit_edge(next_edge, x, next_node)
                if not isinstance(nextx, tp.Tuple):
                    nextx = (nextx,)
                outputs.extend(nextx)
        # else: no more edges

        outputs = tuple(outputs)
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs

    def call(x, *args, **kwargs):
        input_nodes = [
            nodes[0]
            for nodes, edge in tree.edges.items()
            if edge.get("is_input", False)
        ]
        assert len(set(input_nodes)), "multi-inputs not yet supported"
        start_node = input_nodes[0]

        x = [
            visit_edge(next_edge, x, next_node)
            for next_node, next_edge in tree[start_node].items()
        ]
        x = tuple(x)
        if len(x) == 1:
            x = x[0]
        return x

    return call
