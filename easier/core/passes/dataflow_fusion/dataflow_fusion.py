# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import operator
import networkx
from typing import Dict, Iterable, List, Sequence, Set
from torch import fx
import torch

import easier.core.module as esr
from easier.core.passes.dataflow_fusion.node_group import \
    NodeGroup, get_node_group, set_node_group, fuse_groups_if_no_deadlock, \
    get_node_group_connectivity_matrix, \
    set_node_group, get_node_group
from easier.core.passes.dataflow_fusion.debug import \
    dump_visualized_fused_groups
from easier.core.passes.dataflow_fusion.fusion_rules import \
    GroupRule, NodeRule, FuseDecision
import easier.core.passes.dataflow_fusion.fusion_rules as rules
from easier.core.passes.utils import \
    FX, EasierInterpreter, SubmodNameAllocator, get_called_module, tree_map
from easier.core.runtime.metadata import \
    collect_meta, get_node_meta, set_node_meta, StructuredViewSrc, \
    get_node_view_src, set_node_view_src, ViewSrc
from easier.core.utils import logger


class NodeGrouper(EasierInterpreter):
    def __init__(
        self, modules: Sequence[esr.Module], graphs: Sequence[fx.Graph]
    ):
        assert len(modules) == len(graphs) == 1
        super().__init__(modules, graphs)

        self.group_rules: List[GroupRule] = [
            rules.RejectExcludedGroup(),
            rules.ReducerAggregatorGroupsConflict(),
            rules.ReducerInstancesConflict(),
        ]
        self.node_rules: List[NodeRule] = [
            rules.GroupingEndsBeforeSelector(),
            rules.GroupingEndsAfterReducer(),
            rules.GroupingEndsAtReducerOut(),
            rules.GroupingEndsAfterAggregator(),
        ]

        self.visited: Set[fx.Node] = set()

        root = modules[0]
        graph = graphs[0]

        # Init NodeGroup for each Node, as NodeGroup.ctor will calculate
        # downstream NodeGroups, we need to traverse in the reversed list.
        nodes = list(graph.nodes)
        ngs = []
        for id, n in reversed(list(enumerate(nodes))):
            ng = NodeGroup(root, [n], id)
            set_node_group(n, ng)
            ngs.append(ng)  # reversed!
        ngs.reverse()

        self.conn_mat = get_node_group_connectivity_matrix(ngs)

    def _check_rules(self, ng1: NodeGroup, ng2: NodeGroup) -> FuseDecision:
        """
        Check if any rule rejects, in a blacklist manner.

        ng1 is graph-wise upstream to ng2.

        Returns:
        -   REJECT if any rules reject, PASS if all rules pass.
        """
        for group_rule in self.group_rules:
            if group_rule.check(ng1, ng2) == FuseDecision.REJECT:
                return FuseDecision.REJECT

        for n1 in ng1.nodes:
            for n2 in ng2.nodes:
                if n2 in n1.users:
                    for node_rule in self.node_rules:
                        if node_rule.check(n1, n2) == FuseDecision.REJECT:
                            return FuseDecision.REJECT

        return FuseDecision.PASS

    def _try_fuse_downstream_groups(self, node: fx.Node) -> NodeGroup:
        """
        Try to fuse NodeGroup on `node` and its downstream NodeGroups.

        Two NodeGroups are only fused if:
        -   all GroupRules/NodeRules pass;

        -   there won't be _deadlock_ on the potentially fused NodeGroup,
            i.e. on the graph, there cannot be a path going out of that
            NodeGroup and then going back into it.

            Such NodeGroups cannot be codegen-ed, deadlock indicates that
            its input depends on its own output.
        """
        self.visited.add(node)

        ng = get_node_group(node)

        for user in node.users:
            if user not in self.visited:
                downstream_ng = self._try_fuse_downstream_groups(user)
            else:
                downstream_ng = get_node_group(user)

            if ng is not downstream_ng and self._check_rules(
                ng, downstream_ng
            ) == FuseDecision.PASS:
                fuse_groups_if_no_deadlock(ng, downstream_ng, self.conn_mat)

                # If fusion succeeds:
                # - NodeGroup instance `downstream_ng` is discarded;
                # - NodeGroup instance `ng` is in-place expanded;
                # - All Nodes in the fused group are bound to
                #   NodeGroup instance `ng`.

        return ng

    def for_each_node(self):
        if self.current_node not in self.visited:
            self._try_fuse_downstream_groups(self.current_node)


def topo_sort_node_groups(
    ng_conn_mat: torch.Tensor, graph: fx.Graph
) -> List[NodeGroup]:
    ngs: List[NodeGroup] = list(set(map(get_node_group, graph.nodes)))
    ngids = [ng.id for ng in ngs]
    assert len(set(ngids)) == len(ngids)

    ids_tensor = torch.tensor(ngids)
    active_ng_conn_mat = ng_conn_mat[ids_tensor][:, ids_tensor]

    dag = networkx.DiGraph()
    dag.add_nodes_from(range(len(ngids)))

    for rowid, src_ngid in enumerate(ngids):
        for colid in active_ng_conn_mat[rowid].argwhere().ravel().tolist():
            # networkx topo sort disallows self-self edge
            if rowid != colid:
                dag.add_edge(rowid, colid)

    # TODO Alternative: networkx.lexicographical_topological_sort()
    # which can also maintain the original Node order.
    groups = [ngs[rowid] for rowid in networkx.topological_sort(dag)]

    # Since output NG has 0 IO-degrees, it's put in the middle.
    # Because of JitEngine, forward() won't exit early, but better to put
    # it in the correct position, for consistency.
    for topo_i in range(len(groups)):
        output_ng = groups[topo_i]
        if list(output_ng.nodes)[0].op == FX.OUTPUT:
            break
    groups.pop(topo_i)
    groups.append(output_ng)

    return groups


def node_groups_to_graph_modules(
    module: esr.Module, graph: fx.Graph, topo_groups: Iterable[NodeGroup]
):
    """
    Given NodeGroup(nodes=[..., n{j}, ...]) and
    x{i} y{k} that are not in the group:

    Forall xi -> nj
        {..., xi, ...} forms the set of _input nodes_ to this NodeGroup;

    Forall nj -> yk,
        {..., nj, ...} forms the set of _output nodes_ of this NodeGroup.

    The resultant GraphModule will have parameters with names:
        {..., xi.name, ...} of all _input nodes_,
        When calling the GraphModule, bind {..., xi, ...} by their names.

    The resultant GraphModule will have a tuple result, by packing:
        (..., nj, ...) of all _output nodes_.
        After calling the GraphModule, outer Graph should unpack and
        bind the items to variables with names: {..., yk.name, ...}

        NOTE multi-res operator like torch.svd() may be fused into the
        GraphModule, however, generally the tuple-unpacking `getitem()` calls
        will be fused, too. So no more nested structure will appear within
        the result tuple of this GraphModule.

    TODO GraphModule must maintain:
    1.  if previous output is a view derived from an input, new output is still
        a view;
    2.  If an allocator ViewSrc is fused, what will it be after fusion/codegen
        so that JitEngine can still validate it?
    3.  copy Node.meta to new Nodes -- how about dep edges?
    """
    raw_node_ids: Dict[fx.Node, int] = dict(
        (n, i) for i, n in enumerate(graph.nodes)
    )

    outer_g = fx.Graph()
    gm_name_allocator = SubmodNameAllocator('easier')

    # raw Nodes to new, outer Nodes
    outer_node_dict: Dict[fx.Node, fx.Node] = {}

    # new CALL_MODULE Nodes to inner Node dict
    kernels_node_dicts: Dict[fx.Node, Dict[fx.Node, fx.Node]] = {}

    for ng in topo_groups:
        ng: NodeGroup

        if len(ng.nodes) > 1:

            kernel_g = fx.Graph()
            kernel_node_dict: Dict[fx.Node, fx.Node] = {}
            for raw_in in ng.all_input_nodes():
                ph = kernel_g.placeholder(raw_in.name)
                kernel_node_dict[raw_in] = ph

            # Sort the inner Nodes into the original order
            ng_raw_nodes = sorted(ng.nodes, key=raw_node_ids.__getitem__)
            for raw_node in ng_raw_nodes:
                # the 2nd callable `Node -> object` of `.node_copy` will
                # only do the transformation on Node instances,
                # and recursively on the input structures.
                # Other values will be intact.
                kernel_node = kernel_g.node_copy(
                    raw_node, kernel_node_dict.__getitem__
                )
                kernel_node.meta = {}  # node_copy copies meta dict

                kernel_node_dict[raw_node] = kernel_node

            output = []
            for out_node_raw in ng.all_output_nodes():
                out_item = kernel_node_dict[out_node_raw]
                output.append(out_item)

            kernel_g.output(output)

            # Take the same root module to ensure CALL_MODULE on
            # Selectors/Reducers within the kernel reference the instances.
            gm = fx.GraphModule(module, kernel_g)
            set_node_group(gm, ng)

            # e.g. easier3_select_reduce209
            gm_name = gm_name_allocator.alloc_name(
                module, f"_{ng.type.name.lower()}{ng.id}"
            )
            module.add_module(gm_name, gm)

            # Modify the outer graph:
            # - CALL_MODULE `gm_name`
            # - Unpack output item and bind to raw Nodes
            kernel_args = tuple(
                outer_node_dict[raw_in] for raw_in in ng.all_input_nodes()
            )
            call_kernel = outer_g.call_module(gm_name, kernel_args)

            for i, raw_out in enumerate(ng.all_output_nodes()):
                outer_out_item = outer_g.create_node(
                    FX.CALL_FUNCTION, operator.getitem, (call_kernel, i),
                    name=raw_out.name
                )
                outer_node_dict[raw_out] = outer_out_item

            kernels_node_dicts[call_kernel] = kernel_node_dict

        else:
            # Copy excluded or singleton NodeGroups, e.g.:
            # - get_attr
            # - all_gather_into_tensor, HaloExchanger calls
            # - output
            raw_n, = ng.nodes

            # the 2nd callable `Node -> object` of `.node_copy` will
            # only do the transformation on Node instances,
            # and recursively on the input structures.
            # Other values will be intact.
            new_n = outer_g.node_copy(raw_n, outer_node_dict.__getitem__)
            new_n.meta = {}  # node_copy copies meta dict

            outer_node_dict[raw_n] = new_n
        # end if len(ng.nodes) > 1
    # end for ng in groups

    NodeMetaCopier([module], [outer_g], outer_node_dict).run()

    return module, outer_g


class NodeMetaCopier(EasierInterpreter):
    """
    Some kinds of Node.meta in the new, outer Graph
    will be validated by JitEngine.

    Rather than rerun the related passes, after fusing `graph`,
    we'll adjust Node.meta, based on simple rules.
    In this way, we can ensure the semantic consistency before/after
    dataflow fusion.

    Basically, the rules are:

    -   life_range_analysis.get_nodes_end_at()
        -   Not copied.
        -   life_range_analysis relies strictly on the original execution
            order, and we didn't insert such control edges, so life range
            won't be correctly preserved during toposort.
            We have to rerun life_range_analysis after fusion.

    -   ViewSrc(node, index)
        -   If `node` is outside any NodeGroup, its ViewSrc remains unchanged
            (mapped to new Node instacnes).

            Otherwise, the whole ViewSrc is changed to the GraphModel Node
            (the allocator).

            NOTE because all user-programmed views must be clone-ed,
            the inputs and outputs of GraphModule won't be views.

        -   ViewSrc that's totally within the kernel Graph is no longer used.

    -   TensorMeta
        -   Simply copied.

    -   get_data_dependency_inputs()/users()
        -   Not copied.
    """

    def __init__(
        # input graph should be the new, outer graph
        self, modules: Sequence[esr.Module], graphs: Sequence[fx.Graph],
        outer_node_dict: Dict[fx.Node, fx.Node]
    ):
        super().__init__(modules, graphs)

        self.raw2outer = outer_node_dict
        self.outer2raw: Dict[fx.Node, fx.Node] = {
            v: k for k, v in outer_node_dict.items()
        }

        # A ViewSrc is equal to a constant 2-tuple (time, memory_addr),
        # P.S. including time to differentiate reused memory.
        # Given kernel calls will change many original time-addr tuples, we
        # directly maintain the VS-VS mapping between raw and new Graphs.
        self.rawsrc2outersrc: Dict[ViewSrc, ViewSrc] = {}

    def for_each_node(self) -> None:
        handled = super().for_each_node()
        if handled == True:
            # Only certain specialized branches returns True,
            # otherwise it's None or False.
            return

        raw_cur_node = self.outer2raw[self.current_node]

        # Only plain data types like shape/dtype
        set_node_meta(self.current_node, get_node_meta(raw_cur_node))

        def _map_viewsrc(raw_vs):
            if not isinstance(raw_vs, ViewSrc):
                # A general Node may have ViewSrc be None.
                return raw_vs

            if raw_vs.node is raw_cur_node:
                new_src = ViewSrc(self.current_node, raw_vs.index)
                self.rawsrc2outersrc[raw_vs] = new_src
                return new_src
            else:
                # Reflecting the progressively updated new ViewSrc.
                return self.rawsrc2outersrc[raw_vs]

        raw_src: StructuredViewSrc = get_node_view_src(raw_cur_node)
        new_src = tree_map(raw_src, _map_viewsrc)
        set_node_view_src(self.current_node, new_src)

    def if_call_function(self, function) -> bool:
        if function is not operator.getitem:
            return False
        call_kernel = self.current_node.args[0]
        kernel_out_idx = self.current_node.args[1]
        assert isinstance(call_kernel, fx.Node)
        if call_kernel.op != FX.CALL_MODULE:
            return False
        submod = get_called_module(self.current_module, call_kernel)
        if not isinstance(submod, fx.GraphModule):
            return False
        assert isinstance(kernel_out_idx, int)
        ng = get_node_group(submod)

        # Since the CALL_MODULE has been handled, we directly pick
        # element TensorMeta and ViewSrc from it.
        set_node_meta(
            self.current_node,
            get_node_meta(call_kernel)[kernel_out_idx]  # type: ignore
        )

        new_src: ViewSrc = get_node_view_src(
            call_kernel
        )[kernel_out_idx]  # type: ignore
        set_node_view_src(self.current_node, new_src)

        return True

    def if_call_module(self, submod: torch.nn.Module) -> bool:
        if not isinstance(submod, fx.GraphModule):
            return False

        ng = get_node_group(submod)
        assert self.current_node not in self.outer2raw, \
            'Kernel calls are not in the original `outer_node_dict`'

        # concat-ed TensorMeta
        raw_outputs = list(ng.all_output_nodes())

        metas = list(map(get_node_meta, raw_outputs))
        set_node_meta(self.current_node, metas)

        # concat-ed ViewSrc
        output_raw_srcs = collect_meta(
            list(map(get_node_view_src, raw_outputs)),
            leaf_type=ViewSrc
        )
        assert len(output_raw_srcs) == len(set(output_raw_srcs)), \
            'Kernel output should not share memory'

        def _map_kernel_output_viewsrc(out_idx: int, raw_vs: ViewSrc):
            assert raw_vs.node in ng.nodes, \
                'Kernel output memory must be allocated for' \
                ' the operation originally within the fused subgraph'

            # Assume each output represents an invidual new memory
            # OTAH, they cannot be views to each other.
            return ViewSrc(self.current_node, out_idx)

        # NOTE if the NodeGroup has no output, the resultant ViewSrc on
        # CALL_MODULE would be `[]` rather than None -- `[]` aligns with
        # JitEngine.ViewSrcTrackerBase.
        kernel_srcs: List[ViewSrc] = []
        for i, raw_out in enumerate(raw_outputs):
            raw_src = get_node_view_src(raw_out)
            assert isinstance(raw_src, ViewSrc), \
                'Kernel output item shoud not be further nested'

            outer_src: ViewSrc = _map_kernel_output_viewsrc(i, raw_src)
            kernel_srcs.append(outer_src)

            self.rawsrc2outersrc[raw_src] = outer_src

        set_node_view_src(self.current_node, kernel_srcs)

        return True


def fuse_dataflow(modules: List[esr.Module], graphs: List[fx.Graph]):
    assert len(modules) == len(graphs) == 1
    m = modules[0]
    g = graphs[0]

    logger.debug(f'Fusion pass is running for {m.easier_hint_name}')
    grouper = NodeGrouper(modules, graphs).run()
    logger.info(f'Fusion pass has completed for {m.easier_hint_name}')

    if logger.level <= logging.DEBUG:
        dump_visualized_fused_groups(modules, graphs)

    node_groups = topo_sort_node_groups(grouper.conn_mat, g)

    # NOTE NodeGroup methods like all_output_nodes() rely on the real time
    # state of the raw Graph.
    # If the raw Graph `g` is still used elsewhere, don't modify its nodes.
    m, new_g = node_groups_to_graph_modules(m, g, node_groups)

    return [m], [new_g]
