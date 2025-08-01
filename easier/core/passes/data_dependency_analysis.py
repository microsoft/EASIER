# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import dataclasses
import itertools
import operator
from typing import \
    Dict, Iterable, List, Optional, Sequence, TypeAlias, Union, cast

from torch import nn
import torch
from torch.fx.graph import Graph
from torch.fx.node import Node
from easier.core.runtime.metadata import \
    RuntimeTensorMeta, get_node_meta, collect_meta, \
    ViewSrc, get_node_view_src, is_node_skipped

import easier.core.module as esr

from easier.core.passes.utils import \
    FX, EasierInterpreter, OrderedSet, get_called_module, \
    get_dag_connectivity_matrix
from easier.core.runtime.modules import HaloExchanger, all_gather_into_tensor
from easier.core.utils import EasierJitException


KEY__DATA_DEPENDENCY_INPUTS = 'easier_dataDependency_inputs'
KEY__DATA_DEPENDENCY_USERS = 'easier_dataDependency_users'


def get_data_dependency_inputs(node: Node) -> List[Node]:
    # Return a clone to avoid manipulating the node meta dict.
    return list(node.meta.get(KEY__DATA_DEPENDENCY_INPUTS, []))


def get_data_dependency_users(node: Node) -> List[Node]:
    # Return a clone to avoid manipulating the node meta dict.
    return list(node.meta.get(KEY__DATA_DEPENDENCY_USERS, []))


@dataclasses.dataclass(frozen=True, eq=True)
class _DistEnvViewSrc:
    pass


@dataclasses.dataclass(frozen=True, eq=True)
class _ExternalTensorViewSrc:
    tensor: torch.Tensor


_TViewSrc: TypeAlias = Union[ViewSrc, _DistEnvViewSrc, _ExternalTensorViewSrc]


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
        module = modules[0]
        graph = graphs[0]
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
        self.src2writer: Dict[_TViewSrc, Node] = {}

        self.src2readers: Dict[_TViewSrc, OrderedSet[Node]] = {}

        # Key esr.Tensors include:
        # - esr.Tensor referenced by this esr.Module
        # - esr.Tensor only referenced by nested esr.Module, reading/writing
        #   those Tensors will form dependency between nested Module calls.
        et2src: Dict[esr.Tensor, _TViewSrc] = {}

        # Only includes esr.Tensors referenced by this Module.
        nodesrc2et: Dict[ViewSrc, esr.Tensor] = {}

        # Collect ViewSrc for esr.Tensors referenced by this Module first.
        # For such Tensors, we prefer ViewSrc rather than _ExternalViewSrc.
        #
        # In case of get_attr appears later, we need to collect them first.
        #
        # Even esr.Tensor can have aliases, they will have a unique ViewSrc
        # (in JitEngine ViewSrcTracker, we decide the ViewSrc by mem addr)
        class _SelfTensorViewSrcCollector(EasierInterpreter):
            def if_get_attr(self, submod_path: str, attr_name: str, attr_val):
                if is_node_skipped(self.current_module, self.current_node):
                    return
                if isinstance(attr_val, esr.Tensor):
                    etsrc = get_node_view_src(self.current_node)
                    assert isinstance(etsrc, ViewSrc)
                    et2src[attr_val] = etsrc
                    nodesrc2et[etsrc] = attr_val
        _SelfTensorViewSrcCollector(modules, graphs).run()
        self.et2src = et2src
        self.nodesrc2et = nodesrc2et

        # Create connectivity matrix for dataflow only
        self._node2id: Dict[Node, int] = {}
        for nid, n in enumerate(graph.nodes):
            self._node2id[n] = nid
        self.df_conn_mat = get_dag_connectivity_matrix(
            list(graph.nodes), lambda n: n.users, self._node2id.__getitem__
        )

        # A const "ViewSrc" representing the global network state, causing
        # all communication calls to be strictly ordered.
        self._dist_env_viewsrc = _DistEnvViewSrc()

        # =======
        # Runtime information captured for other esr.Modules that call this
        # esr.Module in a nested call stack:

        # esr.Tensors this Module/JitEngine and its all nested Modules
        # read/write.
        # A nested esr.Tensor parameter may be not referenced during execution
        # of this module, so we don't rely on nn.Module.parameters() to avoid
        # complicate the dependency.
        self.read_tensors: OrderedSet[esr.Tensor] = OrderedSet()
        self.write_tensors: OrderedSet[esr.Tensor] = OrderedSet()

        # Whether this Module and its all nested Modules call communication
        # primitives from DistEnv.
        # If called, the calls of this Module in the surrounding/outer
        # esr.Module must be ordered with communication calls there.
        self.called_comm_primitive: bool = False

    def add_data_dependency_edge(self, src: Node, dst: Node):
        # NOTE only existing dataflow/dependency edges are deduplicated,
        # but if the adding edge is the composition of multiple existing edges,
        # we don't deduplicate for such cases.

        if src is dst:
            # inplace ops may be both reader and writer on the same view src.
            return

        if self.df_conn_mat[self._node2id[src], self._node2id[dst]] > 0:
            # Avoid adding a data dependency edge if it's dataflow connected,
            # NOTE by df_conn_mat we already save a lot redundant dep edges,
            # no need to update conn_mat to reflect dep edge connectivity.
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

    def add_reader_dependency(self, arg_src: _TViewSrc):
        """
        Add the current node as a reader to the memory at `arg_src`.
        """
        reader = self.current_node

        latest_writer = self.src2writer.get(arg_src, None)
        if latest_writer is not None:
            # If the Tensor is read-only (args of matmul etc.) it has no
            # previous writers.
            self.add_data_dependency_edge(latest_writer, reader)

        # Other readers in the same barrier-barrier region.
        # Since the next writer will be enforced to be after these readers,
        # between these readers we do not need a order.
        readers = self.src2readers.setdefault(arg_src, OrderedSet())
        readers.add(reader)

        if arg_src in self.nodesrc2et:
            self.read_tensors.add(self.nodesrc2et[arg_src])

    def add_writer_dependency(self, res_src: _TViewSrc):
        """
        Add the current node as a writer on the memory at `res_src`.

        Including the operations that:
        -   allocates the memory -- this is the most common case
        -   in-place modifies the memory, e.g. setitem, Reducer(out=)
        -   purely creates a view, e.g. getitem, reshape
            P.S. without being rule-based it's not that easy to tell if an op
            is purely creating views.
        -   
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

        if res_src in self.nodesrc2et:
            self.write_tensors.add(self.nodesrc2et[res_src])

    def for_each_node(self):
        # Some ops may take multiple input writable tensors, and the result
        # will be a tuple of those tensors themselves.
        #
        # JitEngine also executes and records `operator.setitem` in this way.
        #
        # TODO any FX-traceable torch ops violate this convention? Or any way
        # to detect if the violation happens?
        #
        if is_node_skipped(self.current_module, self.current_node):
            return

        self.arg_srcs: List[_TViewSrc] = collect_meta(
            [
                get_node_view_src(arg)
                for arg in self.current_node.all_input_nodes
                if not is_node_skipped(self.current_module, arg)
            ],
            leaf_type=ViewSrc
        )

        # Handle more specific scenarios like nested esr.Module calls
        # Derived if_xxx handlers may override self.arg_srcs and self.srcs
        # to provide rule-based, specific and simpler decision on dependency.
        self.res_srcs: Optional[List[_TViewSrc]] = None
        super().for_each_node()

        for arg_src in self.arg_srcs:
            self.add_reader_dependency(arg_src)

        # Generally, purely viewing operators like getitem, reshape will
        # return the argument memory addr, making themselves look like
        # writers, and leading to extra writer dep barriers and handling.
        if self.res_srcs is None:
            res_srcs = collect_meta(
                get_node_view_src(self.current_node),
                leaf_type=ViewSrc
            )
        else:
            res_srcs = self.res_srcs
        for res_src in res_srcs:
            self.add_writer_dependency(res_src)

    def _handle_readonly_or_allocation(self):
        """
        Given a Tensor instance and its underlying memory, an operator may
        behave in one or multiple following ways:

        -   view:
            The operator manipulates `Tensor.strides` etc. without touching
            the memory.

        -   read:
            The contents of the memory is read.

        -   write:
            The contents of the memory is modified.

        -   allocate:
            The operator allocates a new memory and a new Tensor.
            Generally followed by a write.

        P.S. the original Tensor-memory binding cannot be changed,
        e.g. `Tensor.set_()` is not supported by EASIER.

        If we were careful enough and checked down to the essentials
        -- which the current implementation of EASIER did not --
        we might find:
        -   `operator.getitem, torch.view` are view-only ops.

            View-only ops are neither read nor write, and are meaningless
            beyond dataflow (where Tensor.strides matter), i.e. the execution
            time of these ops can arbitarily moved around, as long as not
            breaking dataflow.

        -   `torch.reshape` is a view or an allocator.

        -   `easier.runtime.HaloExchanger` is read-only or an allocator.
            (when a HaloExchanger is for Selector and send-only, it returns
            the input Tensor as well as copying some of its contents)

        However, in order to get free from checking operator one-by-one,
        EASIER by default has policies:

        -   The operator is treated as a writer on all memory blocks
            it returns:
            a.  `operator.setitem, torch.fill_` are purely writers;
            b.  `torch.add(a,b), torch.svd()` are allocators,
                writing the newly allocated memory blocks;
            c.  `torch.view` is classfied as writers -- by the default policy.

        -   This method, _handle_readonly_or_allocation(),
                         ********************************
            can be used, in an allow-list manner, to specifically
            mark and handle an operator as a reader or a allocator,
            e.g. `operator.getitem, HaloExchanger`.

        Additionally, for the sake of simplicity, EASIER treats
        views as reads, view-only as read-only.
        For example, if we want to use this method to eliminate the
        "mis-classification" of `torch.view` in the case (c) above,
        we still have to treat it as read-only, rather than pure view.

        TODO Currently handle only getitem and HaloExchanger, since they have
        been identified as special cases for other purposes.
        TODO Handle pure views, just like syntactic construct GET_ATTR.
        """
        res_src = get_node_view_src(self.current_node)
        assert isinstance(res_src, ViewSrc), "Handle uni-res op only"

        if res_src in self.arg_srcs:
            # If one input ViewSrc is returned, the node is read-only
            # !! Override to emphasize the node is read-only
            self.res_srcs = []
        else:
            # The node is an allocator
            self.res_srcs = [res_src]

    def if_get_attr(self, submod_path: str, attr_name: str, attr_val):
        # GET_ATTR is purely syntactic, its position in IR does not matter.
        # NOTE However, GET_ATTR Nodes are still used as ViewSrc.
        self.arg_srcs = []
        self.res_srcs = []

    def if_call_function(self, function) -> None:
        if function is operator.getitem:
            # If getitem is for tuple indexing -- essentially a syntactic
            # construct of EASIER IR,
            # R/W relations only cover that specified item,
            # e.g. svd() returns 3 memory blocks, subsequent dependency edges
            # must be calculated on individual mem, rather than all 3 blocks.
            arg0 = self.current_node.all_input_nodes[0]
            assert not is_node_skipped(self.current_module, arg0)
            is_tuple_indexing = not isinstance(
                get_node_meta(arg0), RuntimeTensorMeta
            )
            if is_tuple_indexing:
                tuple_item_index = self.current_node.args[1]
                assert isinstance(tuple_item_index, int)

                # !! Override self.arg_srcs
                self.arg_srcs = collect_meta(
                    get_node_view_src(arg0)[tuple_item_index],  # type: ignore
                    leaf_type=ViewSrc
                )

            # No matter which kind of getitem, it's a view
            self._handle_readonly_or_allocation()

        elif function is all_gather_into_tensor:
            self.res_srcs = [self._dist_env_viewsrc]

            self.called_comm_primitive = True

    def if_call_module(self, submod: nn.Module):
        """
        `submod` may be:

        -   A nested esr.Module call has no input/output, data dependency
            may occur through:
            -   esr.Tensor instances the inner Module shares/writes.
            -   the inner Module calls communication primitives,
                its call order must be strictly kept.

        -   A HaloExchanger, its call order must be strictly kept.
        """
        from easier.core.runtime.jit_engine.jit_engine import JitEngine

        if isinstance(submod, esr.Module):
            nested_jit_engine = cast(JitEngine, submod.forward.__self__)

            # !! Override
            self.arg_srcs = []
            self.res_srcs = []

            for nested_et_read in nested_jit_engine.read_tensors:
                # Add _ExternalViewSrc only if the Tensor is not referenced
                # by this Module.
                nested_read_src = self.et2src.setdefault(
                    nested_et_read, _ExternalTensorViewSrc(nested_et_read)
                )
                self.arg_srcs.append(nested_read_src)

                # Recursively tell outer Module/JitEngine
                self.read_tensors.add(nested_et_read)

            for nested_et_write in nested_jit_engine.write_tensors:
                nested_write_src = self.et2src.setdefault(
                    nested_et_write, _ExternalTensorViewSrc(nested_et_write)
                )
                self.res_srcs.append(nested_write_src)

                # Recursively tell outer Module/JitEngine
                self.write_tensors.add(nested_et_write)

            # Enforced order of communication
            if nested_jit_engine.called_comm_primitive:
                self.res_srcs.append(self._dist_env_viewsrc)

                self.called_comm_primitive = True

        if isinstance(submod, HaloExchanger):
            # Use default arg_srcs decision.

            # But for res_srcs, HaloExchanger may behave as in-place writing,
            # e.g. when send-only for Selector, it will return the input
            # tensor. By default this will be treated as writing.
            # Let's simplify the dependency by treating HaloXchg as read-only.
            self._handle_readonly_or_allocation()

            # Enforced order of communication
            assert self.res_srcs is not None
            self.res_srcs.append(self._dist_env_viewsrc)

            self.called_comm_primitive = True


class TensorViewUsageChecker(EasierInterpreter):
    """
    NOTE Unless specifically mentioned, when talking about "views" we are
    talking about **multiple** Tensor instances pointing to the same memory.
    Since even a single Tensor is a view.

    Based on standard PyTorch programming model,
    EASIER programming model additionally requires, if an operation returns
    a view of its input, that operation must be followed by,
    immediately and exactly,
    a call to the function `torch.clone()` or the method `Tensor.clone()`.

    TODO maybe `Tensor.to(copy=True)` or whatever function that essentially
    clones.

    This requirement eliminates the cases that:
    -   EASIER programs deal with non-contiguous memory.

        Such memory and access pattern which makes it inefficient
        for CPU cache line, CUDA coalescing memory access etc.

    -   A view Tensor gets accidentally created and written.

        EASIER programs allow in-place Tensor modification, which is less
        common in PyTorch workloads.
        If a view gets accidentally written, it will casue big trouble
        both for users to realize it or for EASIER devs to
        systematically enhance the user experience.

    -   Following passes need to deal with views.

        By enforcing immediate `.clone()`, it leaves no space for extra
        control flow or deadlock in fusion,
        therefore we can consider it an atomic unit of operations
        that creates a new memory.

        If not immediate cloned, it may cause deadlock in fusion,
        and make views to be outputs of NodeGroups, e.g.:
        ```
        view = v.view(...)
        reducer(..., out=v)  # Reducer.out dataflow edge is cut
        clone = view.clone()  # Control-flow depended on reducer, deadlock
        ```

    This subpass covers:

    -   User-program view ops
        (e.g. operator.getitem, torch.view, torch.reshape)
        must be followed by torch/Tensor.clone;

        Including Nodes of both DIST and REPLICA roles.

    -   EASIER-inserted Nodes may return a view, e.g. HaloExchanger,
        as long as it has only one user and the user isn't a view,
        we don't check it.

        Also, it won't be fused.

    -   Inplace ops
        (e.g. operator.setitem, Reducer, torch.add_, Tensor.fill_):
        If they have users, the users must be clone() too.
    """

    def __init__(self, modules: Sequence[esr.Module], graphs: Sequence[Graph]):
        super().__init__(modules, graphs)
        self.visited = set()

    def _check_immediate_user_is_clone(
        self,
        alloctor_or_view: Node,
        input_viewsrcs: List[ViewSrc],
        immediate_node: Node
    ) -> bool:
        """
        If node creates a view, check it's followed by
        only and one and immediate clone().

        Args:
        -   node: generally it's `self.current_node`,
                but for unpacking `getitem` Nodes after a multi-res Node
                it's the multi-res Node itself.
        -   input_viewsrcs: the ViewSrcs that are effectively representing
                input memory addresses to `node`,
                but not always got from `node.all_input_nodes`.
        -   immediate_node: the immediate following Node in the Graph,
                if `self.current_node` has a user, this arg should be the user.
                (not always exactly `self.current_node.next`)

        Returns:
        -   True: the immediate_node is really the correct `.clone()`
        -   False: for valid cases:
            -   this Node is not a view;
            -   this Node does not have any users at all.

        -   raise EasierJitException: for bad cases:
            -   view is not cloned;
            -   not immediately cloned.
        """
        res_viewsrcs: List[ViewSrc] = collect_meta(
            get_node_view_src(alloctor_or_view), leaf_type=ViewSrc
        )
        inplace_res_viewsrcs = set(res_viewsrcs).intersection(input_viewsrcs)
        if len(inplace_res_viewsrcs) == 0:
            return False
        if len(alloctor_or_view.users) == 0:
            return False

        if len(alloctor_or_view.users) == 1:
            user, = alloctor_or_view.users
            if (
                user.op == FX.CALL_FUNCTION and user.target is torch.clone
            ) or (
                user.op == FX.CALL_METHOD and user.target == 'clone'
            ):
                if user is immediate_node:
                    return True

        # TODO Better report which input (may be in nested args) is detected
        raise EasierJitException(
            f"The operation '{alloctor_or_view.format_node()}' in"
            f" '{self.current_module.easier_hint_name}'"
            " returns a view of an input, this operation must be followed"
            " immediately by exactly one `Tensor.clone()` call"
        )

    def for_each_node(self):
        if is_node_skipped(self.current_module, self.current_node):
            # NOTE since we are detecting view based on rumtime info
            # (memory address), if the Node is skipped, we cannot check it.
            return

        if self.current_node.op == FX.CALL_MODULE:
            submod = get_called_module(self.current_module, self.current_node)
            if isinstance(submod, HaloExchanger):
                return

        if self.current_node in self.visited:
            return
        self.visited.add(self.current_node)

        args_viewsrcs: List[ViewSrc] = collect_meta(
            list(map(get_node_view_src, self.current_node.all_input_nodes)),
            leaf_type=ViewSrc
        )

        #
        # Check if a user-programmed view is immediately `.clone()`.
        #
        if not isinstance(
            get_node_meta(self.current_node), RuntimeTensorMeta
        ):
            # Multi-res op is not handled on itself, but handled on its getitem

            # Calls to clone() must be after unpacking getitem Nodes.
            # TODO currently the checks for views in multi-res are simple:
            # - disallowed unnecessary clone() Nodes,
            # - disallowed order of clone() is not the same as getitem()
            immediate_after_unpack = self.current_node.next
            for user in self.current_node.users:
                immediate_after_unpack = immediate_after_unpack.next

            for i, user in enumerate(self.current_node.users):
                assert user.target is operator.getitem
                assert user.args == (self.current_node, i)
                correctly_cloned = self._check_immediate_user_is_clone(
                    user, args_viewsrcs, immediate_after_unpack
                )

                if correctly_cloned:
                    immediate_after_unpack = immediate_after_unpack.next

            self.visited.update(self.current_node.users)

        else:
            self._check_immediate_user_is_clone(
                self.current_node, args_viewsrcs, self.current_node.next
            )


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
    assert len(modules) == len(graphs) == 1, \
        "One module/graph at a time, avoid adding dep edges cross graphs"
    m = modules[0]

    TensorViewUsageChecker(modules, graphs).run()

    dda = DataDependencyAnalyzer(modules, graphs).run()

    from easier.core.runtime.jit_engine.jit_engine import JitEngine
    je = cast(JitEngine, m.forward.__self__)

    je.read_tensors = dda.read_tensors
    je.write_tensors = dda.write_tensors

    je.called_comm_primitive = dda.called_comm_primitive

    return modules, graphs
