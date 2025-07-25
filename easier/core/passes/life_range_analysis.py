# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, List, Sequence

from torch.fx.graph import Graph
from torch.fx.node import Node

import easier.core.module as esr
from easier.core.passes.utils import EasierInterpreter


KEY__NODES_END_HERE = 'easier_lifeRange_nodesEndHere'


def get_nodes_end_at(node: Node) -> List[Node]:
    """
    Get Nodes whose life ranges end at the specified `node`, including:
    -   the argument Nodes of `node` whose last user is `node`;
    -   Node without users end at itself immediately.

    For example, at runtime, after evaluating `node`,
    jit_engine.values.RuntimeValues for those argument Nodes can be freed
    (this does not mean the underlying tensor memory is freed too).
    """
    return node.meta[KEY__NODES_END_HERE]


def set_nodes_end_at(node: Node, ends: List[Node]):
    node.meta[KEY__NODES_END_HERE] = ends


class LifeRangeAnalyzer(EasierInterpreter):
    def __init__(self, modules: Sequence[esr.Module], graphs: Sequence[Graph]):
        super().__init__(modules, graphs)

        self.node2offset: Dict[Node, int] = {}

        for g in graphs:
            for i, n in enumerate(g.nodes):
                assert n not in self.node2offset, \
                    "Nodes won't be shared by multiple Graphs"

                self.node2offset[n] = i

    def for_each_node(self):
        # If no previous for_each_node added on this Node (as a user),
        # still add a [].
        self.current_node.meta.setdefault(KEY__NODES_END_HERE, [])

        if len(self.current_node.users) == 0:
            # no users, life range ends immediately.
            range_end = self.current_node

        else:
            range_end, _offset = max(  # must not be empty list
                [
                    (user, self.node2offset[user])
                    for user in self.current_node.users
                ],
                key=lambda uo: uo[1]
            )

        range_end.meta.setdefault(
            KEY__NODES_END_HERE, []
        ).append(self.current_node)


def analyze_life_range(modules: List[esr.Module], graphs: List[Graph]):
    """
    This pass analyze the _life ranges_ of Nodes.

    After some time point, a Node X may be no longer used in its Graph,
    The range between
    -   the first time that the Node X appears on the Graph
    -   the last Node that the Node X is used as an argument
    is the _life range_ of that Node.
    (if Node X is never used again, the life range ends at itself)


    At runtime, Nodes point to (nested structures of) tensors, forming
    a two-level reference-count relation:

        Node -> torch.Tensor -> tensor's physical memory

    It's beneficial for EASIER to know those Nodes' life ranges,
    so that we can decrement one reference count on the tensor,
    and let Python GC the memory in time.


    Remarkably:
    -   The life range of a Node does not neccessarily match the lifetime of
        the torch.Tensor Python object or the underlying physical memory,
        even if we de-ref the torch.Tensor at the end of the life range:

        -   Many Nodes can point to a single tensor,
            e.g. all inplace ops return the tensor itself;
        -   Many tensors can point to the same memory region, e.g. views.

    -   It can get critical to do life range analysis and de-ref tensors
        in time.
        Because after FX tracing, all original method stackframes in the
        EASIER Python program will flattened and inlined.
        The length of the resultant Graph can get unforeseeably long.

        As Python method-scope GC no longer gets involved at runtime (instead,
        it's JitEngine), with the space complexity O(len(graph)*len(elempart)),
        it easily gets OOM.

        P.S. It's less severe for backend=='none' cases.
    """
    LifeRangeAnalyzer(modules, graphs).run()
    return modules, graphs
