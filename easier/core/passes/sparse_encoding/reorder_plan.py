# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, \
    Type, Union, Callable, cast
from typing_extensions import TypeAlias
import networkx
import more_itertools


import torch
import torch.overrides
from torch import nn
from torch.fx.graph import Graph
from torch.fx.node import Node, Argument, map_arg

from easier.core.runtime.dist_env import get_cpu_dist_env
from easier.core.utils import \
    logger, EasierJitException
import easier.core.module as esr

from easier.core.passes.tensor_grouping import \
    EasierTensorGroup, get_tensor_groups_relation
from easier.core.passes.utils import \
    EasierInterpreter, OrderedSet


class ReorderGraphBuilder(EasierInterpreter):
    """
    Reordering graph is a cyclic directed graph
    -   node: TensorGroup
    -   edge: layout of which TensorGroup affects other TensorGroups
    """

    def __init__(self, modules: Sequence[esr.Module], graphs: Sequence[Graph]):
        super().__init__(modules, graphs)

        # Not multi-edge.
        # An edge x->y means on the dataflow there is `x=reducer(y)`.
        #
        # One TensorGroup has at most one downstream Reducer.
        self.reducer_edges: Dict[
            Tuple[EasierTensorGroup, EasierTensorGroup],
            esr.Reducer
        ] = {}

        # How many times a Reducer is called on IR.
        self.reducer_nnodes: Dict[esr.Reducer, int] = {}

        # An edge x->y means on the dataflow there is `y=selector(x)`.
        #
        # However, it's multi-edge and Selectors conflict may exist, e.g.
        # A-S1->B, A->S2->B
        # because we don't remove Selector conflicts like bind_reducer.
        self.selector_edges: Dict[
            Tuple[EasierTensorGroup, EasierTensorGroup],
            OrderedSet[esr.Selector]
        ] = {}

        # How many times a Selector is called on IR.
        # The number of IR Nodes is used, as those Selector will always have
        # the same length of idx vectors, just like bind_reducer; and always
        # be into the same TensorGroup.
        self.selector_nnodes: Dict[esr.Selector, int] = {}

    def if_call_module(self, submod: nn.Module) -> None:

        if isinstance(submod, esr.Reducer):
            reducer = submod

            dataflow_input_grp, dataflow_output_grp = \
                get_tensor_groups_relation(self.current_node, reducer)

            key = (dataflow_output_grp, dataflow_input_grp)  # reverse dataflow
            self.reducer_edges[key] = reducer

            self.reducer_nnodes[reducer] = \
                self.reducer_nnodes.get(reducer, 0) + 1

        elif isinstance(submod, esr.Selector):
            selector = submod

            dataflow_input_grp, dataflow_output_grp = \
                get_tensor_groups_relation(self.current_node, selector)

            # We record the Selector, regardlessly of having concat or not.
            # Unlike that we discard concat-ed tensor for Reducer,
            # for Selector we reorder output based on input, so the output can
            # possibly follow the whatever order of the concat-ed tensor
            # (if there is no Selector conflict at node-level).
            key = (dataflow_input_grp, dataflow_output_grp)
            multiedges = self.selector_edges.setdefault(key, OrderedSet())
            multiedges.add(selector)

            self.selector_nnodes[selector] = \
                self.selector_nnodes.get(selector, 0) + 1


def build_cascade_reorder_plan_on_rank0(graph_builder: ReorderGraphBuilder):
    """
    The utmost goal of this pass is to help and provide guidance to
    reorder elements in TensorGroups, so that the reduction computation
    represented by Reducers can:
    - sequentially and continuously read elements to accumulate;
    - sequentially write the memory with the reduction results.

    The idea is to consider such a directed graph, formed by TensorGroups
    and Reducer/Selector instances, and graph edges represent the dependency
    the reordering of TensorGroups.

    However, taking the most general EASIER programs into consideration,
    there are some exceptions:
    1.  the graph may be cyclic, e.g. A-R1->B, B-R2->A

        Solution: we prioritize the Reducer with longer `.idx` in the cycle,
        this is a heuristic rule for more compilicated Reducers.
        And we can break up cycles with the objective of maximizing total
        `.idx` lengths.
        And we prefer to preserve Reducers, so, if there is a Selector,
        we break up cycles at the Selector edge first.

    2.  the TensorGroup may not be directly Reduced, e.g.
        halos are concat-ed and then reduced

        Solution: We try to partially reorder the input TensorGroup.

    NOTE because we need to handle TensorGroups in a collective order among
    nodes, we first decide a global graph and replicate it to all nodes.
    Then we do node-specific reordering.

    Last, we need to ensure local `ElemPart.idx` are well bound to esr.Tensors.

    On the global reorder graph:
    1.  Add all Reducer edges and Selector edges to the (multi-edge) graph;

    2.  Remove conflict inwards edges, which are inwards edges into the same
        TensorGroup.
        We choose the edge to keep using the heuristic rule:
        2.a.    Reducer, at most one instance
        2.b.    if there is no Reducer, choose the Selector which
                has most `call_module` Nodes in the IR

        Now, the TensorGroup reordering can be now uniquely decided;

    3.  Break directed cycles, on:
        3.a.    Selector with minimum `idx_length * node_number`
        3.b.    if there's no Selector, choose the Reducer with
                minimum `(is_full, idx_length * node_number)`

        Now, the TensorGroup reordering inference will not be deadlocked;

    4.  Toposort the TensorGroups on the graph, the graph they form
        -- which is actually a tree -- will be the order among which we
        ** cascade reorder ** those TensorGroups.
    """

    ######### WARNING ##########
    #
    #   `networkx.DiGraph` internal data (e.g. how edges are stored) are
    #   not deterministic or replicated-across-nodes.
    #
    #   ## The non-collective region begins ##
    #
    ######### WARNING ##########

    # NOTE With networkx.MultiDiGraph, essentially an edge is identified with
    # the triple `(srcgrp:TensorGroup, dstgrp:TensorGroup, key:int)`.
    # When dealing with identities of edges and edge-related APIs of networkx,
    # we need to keep especially the keys consistent.
    # Caveat: networkx.MDG APIs often allow omit the key, and simply return
    # the first inserted edge between those endpoints.
    reorder_graph = networkx.MultiDiGraph()

    # Step 1: add all potential edges to the graph
    for (depsrc, depdst), reducer in \
            graph_builder.reducer_edges.items():
        reorder_graph.add_edge(
            depsrc, depdst,
            **{'pattern': reducer}
        )

    for (depsrc, depdst), conflict_selectors in \
            graph_builder.selector_edges.items():
        for selector in conflict_selectors:
            reorder_graph.add_edge(
                depsrc, depdst,
                **{'pattern': selector}
            )

    # Step 2: remove multiple inwards edges
    initial_in_degrees: List[Tuple[EasierTensorGroup, int]] = list(
        reorder_graph.in_degree)
    for into_grp, initial_in_degree in initial_in_degrees:
        # We record `initial_in_degrees` to skip conflict-free TensorGroups,
        # but at some iteration, a recorded TensorGroups that initially
        # has conflict may turn to no longer have conflict,
        # this is because edges to that TensorGroup may have all been
        # removed during the for-in loop.
        if initial_in_degree > 1:

            inwards_reducer: Optional[Tuple[
                #   depsrc,            depdst,        multiedgekey
                EasierTensorGroup, EasierTensorGroup, int
            ]] = None
            inwards_weighted_selectors: List[Tuple[
                #   depsrc,            depdst,   multiedgekey, nnodes
                EasierTensorGroup, EasierTensorGroup, int, int
            ]] = []

            # Create a copy in case of read-during-modification.
            in_edges: List[
                Tuple[EasierTensorGroup, EasierTensorGroup, int, dict]
            ] = list(reorder_graph.in_edges(into_grp, keys=True, data=True))

            for depsrc, depdst, multi_edge_key, edge_data in in_edges:
                pattern = cast(Union[esr.Reducer, esr.Selector],
                               edge_data['pattern'])
                if isinstance(pattern, esr.Selector):
                    nnodes = graph_builder.selector_nnodes[pattern]
                    inwards_weighted_selectors.append(
                        (depsrc, depdst, multi_edge_key, nnodes))

                elif isinstance(pattern, esr.Reducer):
                    assert inwards_reducer is None, \
                        "After bind_reducer," \
                        " conflict Reducer is not expected to occur"
                    inwards_reducer = (depsrc, depdst, multi_edge_key)

                else:
                    assert False, "Must be a Selector or Reducer"

            # Pick the most preferred edge to keep:
            # - if there is a Reducer, pick it;
            # - must have some Selectors, pick the one with the longest idx vec
            if inwards_reducer is not None:
                kept_edge = inwards_reducer
            else:
                kept_edge = max(
                    inwards_weighted_selectors,
                    key=lambda src_dst_key_nnodes: src_dst_key_nnodes[3]
                )[:3]

            for depsrc, depdst, multi_edge_key, edge_data in in_edges:
                if (depsrc, depdst, multi_edge_key) != kept_edge:
                    reorder_graph.remove_edge(depsrc, depdst, multi_edge_key)

    # Step 3: break cycles
    while True:
        # nx.simple_cycles dynamically yields all cycles, we pick at most one
        # then stop generating/calculating more results.
        #
        # P.S. nx.simple_cycles, if enumerated, will cover all cycles. E.g.
        # in the graph A<->B<->C<->A, two cycles A-B and A-B-C will be yielded,
        # i.e. edges may be duplicated.
        r_s_cycles: List[List[EasierTensorGroup]] = more_itertools.take(
            1,
            networkx.simple_cycles(reorder_graph)
        )
        if len(r_s_cycles) == 0:
            break

        cyc0 = r_s_cycles[0]
        # e.g. [grp1, gtp2, grp3] means g1->g2->g3->g1. Directed.
        # Self-edge like grp1->grp1 is removed too, it's not reorder-able.
        #
        # And it doesn't contain edge keys.
        # However, after removal of conflict edges, between depsrc-depdst
        # there is now only one edge, we don't need to pass a key to networkx
        # APIs.
        # NOTE the key of the left edge is not always 0!

        cyc0_hints = []  # logging-only

        cyc0_weighted_selectors: List[
            Tuple[
                # idxlen*nnodes
                Tuple[int],
                EasierTensorGroup, EasierTensorGroup
            ]
        ] = []

        cyc0_weighted_reducers: List[
            Tuple[
                # fullness,  idxlen*nnodes,
                Tuple[float, int],
                EasierTensorGroup, EasierTensorGroup
            ]
        ] = []

        # `windowed([a,b,c]+[a],2)` returns `[(a,b),(b,c),(c,a)]` so that
        # we get all edges on the cycle, given the cycle data format above.
        for (depsrc, depdst) in more_itertools.windowed(cyc0 + [cyc0[0]], 2):
            assert isinstance(depsrc, EasierTensorGroup)
            assert isinstance(depdst, EasierTensorGroup)

            [(multi_edge_key, edge_data)] = \
                list(reorder_graph[depsrc][depdst].items())
            pattern = cast(
                Union[esr.Reducer, esr.Selector], edge_data['pattern']
            )
            idxlen = pattern.easier_data_loader.shape[0]

            if isinstance(pattern, esr.Selector):
                nnodes = graph_builder.selector_nnodes[pattern]
                cyc0_weighted_selectors.append((
                    (idxlen * nnodes,),
                    depsrc, depdst
                ))
            elif isinstance(pattern, esr.Reducer):
                nnodes = graph_builder.reducer_nnodes[pattern]
                cyc0_weighted_reducers.append((
                    (
                        # fullness = len(unique(R.idx)) / R.n
                        float(
                            pattern.easier_data_loader.count_unique()
                        ) / pattern.n,
                        idxlen * nnodes,
                    ),
                    depsrc, depdst
                ))
            else:
                assert False, "Must be a Selector or Reducer"

            cyc0_hints.append(f"{depsrc.hint} -" + (  # type: ignore
                "R" if isinstance(pattern, esr.Reducer) else "S"
            ) + "->")

        # the 1st tuple component e.g. Tuple[bool,int] is compared, and Python
        # tuples are ordered lexicographically.
        if len(cyc0_weighted_selectors) > 0:
            (_minweights, minsrc, mindst) = min(
                cyc0_weighted_selectors, key=lambda tp: tp[0]
            )
        else:
            (_minweights, minsrc, mindst) = min(
                cyc0_weighted_reducers, key=lambda tp: tp[0]
            )

        cyc0_hint = ' '.join(cyc0_hints)
        breaking_hint = ' -> '.join([minsrc.hint, mindst.hint])
        logger.debug(
            f"Breaking reordering cycle [{cyc0_hint}] on {breaking_hint}"
        )

        reorder_graph.remove_edge(minsrc, mindst, key=None)  # ok to omit key

    # Step 4: toposort

    # dfs_tree is logging-only, we rely on toposort later.
    cascade_hints = []
    for (hintsrc, hintdst) in (networkx.dfs_tree(reorder_graph).edges()):
        [(multi_edge_key, edge_data)] = list(
            reorder_graph[hintsrc][hintdst].items()
        )
        pattern = edge_data['pattern']
        cascade_hints.append(f"{hintsrc.hint} -" + (
            "R" if isinstance(pattern, esr.Reducer) else "S"
        ) + f"-> {hintdst.hint}")
    logger.debug(
        "Cascade-reordering tree:\n\t" + '\n\t'.join(cascade_hints)
    )

    # Will include isolated TensorGroup whose edges are all removed;
    # but does not include TensorGroup without comm at all.
    to_reorders: List[EasierTensorGroup] = list(
        networkx.topological_sort(reorder_graph)
    )

    ## The non-collective region ends ##

    plan: List[CascadeReorderStep] = []
    for to_reorder in to_reorders:
        upstreams: List[Tuple[EasierTensorGroup, EasierTensorGroup, dict]] \
            = list(reorder_graph.in_edges(to_reorder, data=True))
        if len(upstreams) == 0:
            # Because `remove_edge` will not automatically remove the node
            # if the in/out-degree reach 0, we could skip the TensorGroup
            # that happens to lose all edges.
            pass  # no op
        elif len(upstreams) == 1:
            (depsrc, depdst, edge_data) = upstreams[0]
            pattern = cast(
                Union[esr.Reducer, esr.Selector], edge_data['pattern']
            )

            step = CascadeReorderStep(
                target=depdst, pattern=pattern, basis=depsrc
            )
            plan.append(step)

        else:
            assert len(upstreams) <= 1, \
                "Each TensorGroup must have <=1 inwards reordering edges"

    return plan


@dataclass
class CascadeReorderStep:
    target: EasierTensorGroup
    pattern: Union[esr.Selector, esr.Reducer]
    basis: EasierTensorGroup


def broadcast_plan(
    plan: List[CascadeReorderStep],
    graph_builder: ReorderGraphBuilder
) -> List[CascadeReorderStep]:
    dist_env = get_cpu_dist_env()

    groups: Dict[EasierTensorGroup, int] = {}
    patterns: Dict[Union[esr.Reducer, esr.Selector], int] = {}

    # get replicated IDs for TensorGroup and Reducer/Selector based on
    # object structures in `graph_builder`

    nreducers = len(graph_builder.reducer_edges)
    for i, ((src, dst), reducer) in \
            enumerate(graph_builder.reducer_edges.items()):
        isrc = i * 2
        idst = i * 2 + 1
        groups[src] = isrc
        groups[dst] = idst
        patterns[reducer] = i

    if len(graph_builder.selector_edges) > 0:
        nseledges = max(len(oset)
                        for oset in graph_builder.selector_edges.values())
        for i, ((src, dst), selectors) in \
                enumerate(graph_builder.selector_edges.items()):
            isrc = 2 * nreducers + i * 2
            idst = 2 * nreducers + i * 2 + 1
            groups[src] = isrc
            groups[dst] = idst

            for j, selector in enumerate(selectors):
                patterns[selector] = nreducers + i * nseledges + j

    rgroups = dict((v, k) for k, v in groups.items())
    rpatterns = dict((v, k) for k, v in patterns.items())

    assert len(rgroups) == len(groups)
    assert len(rpatterns) == len(patterns)

    if dist_env.rank == 0:
        target_ids = [groups[step.target] for step in plan]
        basis_ids = [groups[step.basis] for step in plan]
        pattern_ids = [patterns[step.pattern] for step in plan]

        dist_env.broadcast_object_list(0, [target_ids, basis_ids, pattern_ids])

    else:
        [target_ids, basis_ids, pattern_ids] = \
            dist_env.broadcast_object_list(0)

        plan = [
            CascadeReorderStep(
                target=rgroups[tid], pattern=rpatterns[pid], basis=rgroups[bid]
            ) for tid, bid, pid in zip(target_ids, basis_ids, pattern_ids)
        ]

    return plan


def build_cascade_reorder_plan(
    modules: List[esr.Module], graphs: List[Graph]
) -> List[CascadeReorderStep]:
    graph_builder = ReorderGraphBuilder(modules, graphs)
    graph_builder.run()

    dist_env = get_cpu_dist_env()
    if dist_env.rank == 0:
        plan = build_cascade_reorder_plan_on_rank0(graph_builder)

        # `reorder_xxx_by_xxx` functions are collective, so we need to ensure
        # the plan list is replicated among nodes.
        plan = broadcast_plan(plan, graph_builder)
    else:
        plan = broadcast_plan([], graph_builder)

    return plan
