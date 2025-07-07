# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import itertools
import operator
from typing import Dict, Iterable, List, Optional, Sequence

from torch import nn
from torch.fx.graph import Graph
from torch.fx.node import Node
from easier.core.runtime.metadata import \
    Role, RuntimeTensorMeta, get_node_meta, collect_meta, \
    ViewSrc, get_node_view_src

import easier.core.module as esr

from easier.core.passes.utils import \
    FX, EasierInterpreter, OrderedSet


KEY__DATA_DEPENDENCY_INPUTS = 'easier_dataDependency_inputs'
KEY__DATA_DEPENDENCY_USERS = 'easier_dataDependency_users'


def get_data_dependency_inputs(node: Node) -> List[Node]:
    # Return a clone to avoid manipulating the node meta dict.
    return list(node.meta.get(KEY__DATA_DEPENDENCY_INPUTS, []))


def get_data_dependency_users(node: Node) -> List[Node]:
    # Return a clone to avoid manipulating the node meta dict.
    return list(node.meta.get(KEY__DATA_DEPENDENCY_USERS, []))


class DataDependencyAnalyzer(EasierInterpreter[None]):
    """
    When a Node/tensor X is used as an argument to another Node/operation F,
    we say F does a reading operation on the memory under X;
    similarily, when a argument Node/tensor Y is inplace modified by F,
    or the operation F allocates the memory of Y,
    we say F does a writing operation on the memory under Y.

    Along the node list as the original EASIER program runs, at different
    timings/Nodes, a memory may be modified (and maybe multiple times!).
    Those modifying timings/Nodes become barriers.
    Any operation/Node that directly reads/writes a memory, cannot be
    reordered across any barrier formed by specifically that memory.
    (i.e. crossing barrier of other memory does not matter.)

    This Analyzer will add data dependency edges between those reader/writers
    Nodes to express such barrier-like constraints.

    We present a memory by the tensor's ViewSrc
    i.e. ignoring strides/offsets introduced by the PyTorch view operations
    e.g. X[:, :, 2].
    """

    def __init__(self, modules: Sequence[esr.Module], graphs: Sequence[Graph]):
        assert len(modules) == len(graphs) == 1, \
            "One module/graph at a time, avoid adding dep edges cross graphs"
        # otherwise, 1) provide multiple stackframes,
        # 2) clear add2readers/write when switching current_graph.

        super().__init__(modules, graphs)

        # Each writer Node refreshes the status of a view source;
        # All subsequent readers, and the next writer, have
        # data dependency on this writer.
        # Including the Node that is the operation who allocates the memory.
        #
        # When a writer Node (e.g. Reducer-with-out) is never referred
        # in dataflow, it will still be added to src2writer,
        # but may not be touched anymore, e.g. no more writers/readers torch
        # the same view source of `out=x`.
        self.src2writer: Dict[ViewSrc, Node] = {}

        self.src2readers: Dict[ViewSrc, OrderedSet[Node]] = {}

    def add_data_dependency_edge(self, src: Node, dst: Node):
        # NOTE only existing dataflow/dependency edges are deduplicated,
        # but if the adding edge is the composition of multiple existing edges,
        # we don't deduplicate for such cases.

        if src is dst:
            # inplace ops may be both reader and writer on the same view src.
            return

        if src in dst.all_input_nodes:
            # Avoid adding a data dependency edge if a dataflow edge
            # already exists (between the two Nodes).
            return

        # Use OrderedSet to deduplicate.
        dep_inputs: OrderedSet[Node] = dst.meta.setdefault(
            KEY__DATA_DEPENDENCY_INPUTS, OrderedSet()
        )
        dep_users: OrderedSet[Node] = src.meta.setdefault(
            KEY__DATA_DEPENDENCY_USERS, OrderedSet()
        )

        # Since we traverse each Node in the node list once, and have avoid
        # above cases, we won't add redundant data dependency edges between
        # the same (src,dst)-pair.

        dep_inputs.add(src)
        dep_users.add(dst)

    def add_reader_dependency(self, arg_src: ViewSrc):
        """
        Add the current node as a reader to the memory at `arg_src`.
        """
        reader = self.current_node

        latest_writer = self.src2writer[arg_src]
        self.add_data_dependency_edge(latest_writer, reader)

        # Other readers in the same barrier-barrier region.
        # Since the next writer will be enforced to be after these readers,
        # between these readers we do not need a order.
        readers = self.src2readers.setdefault(arg_src, OrderedSet())
        readers.add(reader)

    def add_writer_dependency(self, res_src: ViewSrc):
        """
        Add the current node as a writer on the memory at `res_src`.

        Including the operations that:
        -   allocates the memory -- this is the most common case
        -   in-place modifies the memory, e.g. setitem, Reducer(out=)
        -   purely creates a view, e.g. getitem, reshape
            P.S. without being rule-based it's not that easy to tell if an op
            is purely creating views.
        """
        writer = self.current_node

        # If current_node is the operation that allocates, we get no prev
        # writers.
        prev_writer: Optional[Node] = self.src2writer.get(res_src, None)
        prev_readers: Iterable[Node] = self.src2readers.get(res_src, [])

        # By concat-ing prev_write with prev_reads, we add a dedicated dep edge
        # between the two writes, even there are reads between them, e.g.
        # ```
        # n1 = setitem(a, (:), 42)
        # b  = getitem(a, (:))
        # n2 = setitem(a, (:), 43)
        # ```
        # we get 3 dep edges `n1->b, b->n2, n1->n2` even though the first 2
        # dep edges are composable.

        for prev in itertools.chain([prev_writer], prev_readers):
            if prev is None:
                continue  # may have no prev write

            # We will avoid adding a data dependency edge if a dataflow edge
            # already exists (between the two Nodes).
            # But we do not bother checking
            # if a dataflow path (i.e. many connected edges) exists.
            self.add_data_dependency_edge(prev, writer)

        # refresh the status of the view src ("add a barrier")
        self.src2writer[res_src] = writer
        self.src2readers[res_src] = OrderedSet()

    def is_skipped(self, node: Node):
        meta = get_node_meta(node)
        if isinstance(meta, RuntimeTensorMeta):
            return meta.role == Role.DISTRIBUTED and meta.shape[0] == 0
        return False

    def for_each_node(self):
        # Some ops may take multiple input writable tensors, and the result
        # will be a tuple of those tensors themselves.
        #
        # JitEngine also executes and records `operator.setitem` in this way.
        #
        # TODO any FX-traceable torch ops violate this convention? Or any way
        # to detect if the violation happens?
        #
        if self.is_skipped(self.current_node):
            return

        arg_srcs = None  # type: ignore

        # If getitem is for tuple indexing, which is essentially a syntactic
        # construct of EASIER IR, R/W relations only cover that specified item.
        if self.current_node.target == operator.getitem:
            arg0 = self.current_node.all_input_nodes[0]
            assert not self.is_skipped(arg0)
            is_tuple_indexing = not isinstance(
                get_node_meta(arg0), RuntimeTensorMeta
            )
            if is_tuple_indexing:
                tuple_item_index = self.current_node.args[1]
                assert isinstance(tuple_item_index, int)
                arg_srcs = collect_meta(
                    get_node_view_src(arg0)[tuple_item_index],  # type: ignore
                    leaf_type=ViewSrc
                )

        # For all other cases than tuple indexing:
        if arg_srcs is None:
            arg_srcs = collect_meta(
                [
                    get_node_view_src(arg)
                    for arg in self.current_node.all_input_nodes
                    if not self.is_skipped(arg)
                ],
                leaf_type=ViewSrc
            )

        arg_srcs: List[ViewSrc]
        for arg_src in arg_srcs:
            self.add_reader_dependency(arg_src)

        # Generally, purely viewing operators like getitem, reshape will
        # return the argument memory addr, making themselves look like
        # writers, and leading to extra writer dep barriers and handling.
        res_srcs = collect_meta(
            get_node_view_src(self.current_node),
            leaf_type=ViewSrc
        )
        for res_src in res_srcs:
            self.add_writer_dependency(res_src)

        # Handle more specific scenarios like nested esr.Module calls
        super().for_each_node()

    # TODO
    # Some functions are purely manipulating the views, e.g.
    # `a2 = a[...]; a3 = a[...]`, and followed by `a[:] = 42; b = sin(a2)`.
    # As long as we ensure `a[:] and b` are kept in order,
    # it's actually safe to move `a2 a3` around.
    # We might omit adding data dependency edges on them if needed.

    def if_call_module(self, submod: nn.Module):
        if isinstance(submod, esr.Module):
            # A nested esr.Module call has no input/output, data dependency
            # may occur through esr.Tensor instances the inner Module
            # shares/writes.
            # TODO we may identify the concrete RECURSIVE intersection set
            # of esr.Tensors:
            # intersect(tensors, union(read_write_tensors for nested_mod))

            #
            # For the sake of simplicity,
            # add read/write dependencies on ALL esr.Tensors in current_module.
            #
            param_srcs_to_write: OrderedSet[ViewSrc] = OrderedSet()
            # However, get_attr Nodes may appear after the esr.Module call,
            # in such cases the initial writer for the ViewSrc hasn't
            # been decided yet -- the esr.Module call is the initial
            # writer.
            param_srcs_to_read: OrderedSet[ViewSrc] = OrderedSet()

            before_call_node = True
            for getattr in self.current_graph.nodes:
                if getattr is self.current_node:
                    before_call_node = False

                if getattr.op == FX.GET_ATTR:
                    if self.is_skipped(getattr):
                        # Even it's get_attr of an esr.Tensor, if it's skipped
                        # the Node has no view info.
                        continue

                    attr_view_src = get_node_view_src(getattr)
                    assert isinstance(attr_view_src, ViewSrc)
                    param_srcs_to_write.add(attr_view_src)

                    if before_call_node:
                        param_srcs_to_read.add(attr_view_src)

            for param_src in param_srcs_to_read:
                self.add_reader_dependency(param_src)

            for param_src in param_srcs_to_write:
                self.add_writer_dependency(param_src)


def analyze_data_dependency(modules: List[esr.Module], graphs: List[Graph]):
    """
    PyTorch operations that take Nodes/tensors as inputs and return a
    Node(itself)/tensor, are actually reading/writing the storage/memory
    underneath those Nodes/tensors.

    The idea of "views" informs that different Nodes/tensors may refer to the
    same storage, therefore the reading/writing ops should follow the order
    of `graph.nodes: List[Node]` when we reorder the node list, even though
    thinking nodes to form a dataflow graph, in a graph-theoretical sense.

    The data dependency analysis enforces such a reading/writing order by
    adding _data dependency edges_ to the graph.

    Graph manipulation passes (like dataflow fusion) can leverage those extra
    edges (between data dependency inputs/users) as well as the original
    dataflow edges (between node inputs/users) to ensure numerical correctness
    with tensor-writing operations like `y[:]=x` or `Reducer.forward(out=x)`.
    """
    DataDependencyAnalyzer(modules, graphs).run()

    return modules, graphs
