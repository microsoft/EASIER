# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, List, Optional, Sequence

from torch.fx.graph import Graph
from torch.fx.node import Node

import easier.core.module as esr
from easier.core.passes.utils import EasierInterpreter


KEY__LAST_USER = 'easier_lifeRange_lastUser'
KEY__NODES_END_HERE = 'easier_lifeRange_nodesEndHere'


def get_last_user(node: Node) -> Optional[Node]:
    """
    Get the last user Node for the input `node`.
    If `node` has no user, returns None.
    """
    return node.meta[KEY__LAST_USER]


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
        # If no previous for_each_node added LAST_USES, still add a [].
        self.current_node.meta.setdefault(KEY__NODES_END_HERE, [])

        if len(self.current_node.users) == 0:
            self.current_node.meta[KEY__LAST_USER] = None

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

            self.current_node.meta[KEY__LAST_USER] = range_end

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
