# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import operator
import networkx
from typing import Dict, List, Sequence, Set
import scipy
from torch import fx
import torch

import easier.core.module as esr
from easier.core.passes.dataflow_fusion.node_group import \
    NodeGroup, get_node_group, set_node_group, fuse_groups_if_no_deadlock, \
    get_node_group_connectivity_matrix, \
    set_graph_module_node_group, get_graph_module_node_group
from easier.core.passes.dataflow_fusion.debug import \
    dump_visualized_fused_groups
from easier.core.passes.dataflow_fusion.fusion_rules import \
    GroupRule, NodeRule, FuseDecision
import easier.core.passes.dataflow_fusion.fusion_rules as rules
from easier.core.passes.utils import FX, EasierInterpreter, SubmodNameAllocator, tree_map
from easier.core.utils import logger
from easier.core.runtime.dist_env import get_runtime_dist_env, unbalanced_compute_heartbeat


class NodeGrouper(EasierInterpreter):
    def __init__(self, modules: Sequence[esr.Module], graphs: Sequence[fx.Graph]):
        assert len(modules) == len(graphs) == 1
        super().__init__(modules, graphs)

        self.group_rules: List[GroupRule] = [
            rules.RejectExcludedGroup(),
            rules.ReducerAggregatorGroupsConflict(),
            rules.ReducerInstancesConflict(),
            rules.UnifyReducerScheduling(),
        ]
        self.node_rules: List[NodeRule] = [
            rules.GroupingEndsBeforeSelector(),
            rules.GroupingEndsAfterReducer(),
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


def node_groups_to_graph_modules(
    ng_conn_mat: torch.Tensor, module: esr.Module, graph: fx.Graph
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

    outer_g = fx.Graph()
    gm_name_allocator = SubmodNameAllocator('easier')
    outer_node_dict: Dict[fx.Node, fx.Node] = {}

    for ng in groups:
        ng: NodeGroup


        if len(ng.nodes) > 1:

            kernel_g = fx.Graph()
            kernel_node_dict: Dict[fx.Node, fx.Node] = {}
            for raw_in in ng.all_input_nodes():
                ph = kernel_g.placeholder(raw_in.name)
                kernel_node_dict[raw_in] = ph
            
            ng_raw_nodes = sorted(ng.nodes, key=raw_node_ids.__getitem__)
            for raw_node in ng_raw_nodes:
                # the 2nd callable `Node -> object` of `.node_copy` will
                # only do the transformation on Node instances,
                # and recursively on the input structures.
                # Other values will be intact.
                kernel_node = kernel_g.node_copy(
                    raw_node, kernel_node_dict.__getitem__
                )
                kernel_node_dict[raw_node] = kernel_node
            
            output = []
            for out_node_raw in ng.all_output_nodes():
                out_item = kernel_node_dict[out_node_raw]
                output.append(out_item)
            
            kernel_g.output(output)

            # Take the same root module to ensure CALL_MODULE on
            # Selectors/Reducers within the kernel reference the instances.
            gm = fx.GraphModule(module, kernel_g)
            set_graph_module_node_group(gm, ng)

            # e.g. easier3_select_reduce
            gm_name = gm_name_allocator.alloc_name(
                module, f"_{ng.type.name.lower()}"
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

        else:
            raw_n, = ng.nodes

            # the 2nd callable `Node -> object` of `.node_copy` will
            # only do the transformation on Node instances,
            # and recursively on the input structures.
            # Other values will be intact.
            new_n = outer_g.node_copy(raw_n, outer_node_dict.__getitem__)
            outer_node_dict[raw_n] = new_n
        # end if len(ng.nodes) > 1
    # end for ng in groups

    outer_g.output(None)

    NodeMetaCopier([module], [outer_g], outer_node_dict).run()

    return module, outer_g


class NodeMetaCopier(EasierInterpreter):
    """
    Some kinds of Node.meta will be validated by JitEngine,
    after fusing `graph`, we'll use simple rule-based methods to adjust
    Node.meta to ensure the semantic consistency before/after dataflow fusion.
    (Rather than rerun the related passes.)

    life_range_analysis.get_nodes_end_at()
    -   On CALL_MODULE, U{ get_nodes_end_at(ng_n) for ng_n in ng.nodes }.

    ViewSrc(node, index)
    -   If `node` is outside any NodeGroup, it remains (mapped to new Nodes).
        Otherwise, the whole ViewSrc is changed to the GraphModel (allocator).
    
    TensorMeta
    -   Simply copy.

    get_data_dependency_inputs()/users()
    -   Not copied.
    """
    def __init__(
        self, modules: Sequence[esr.Module], graphs: Sequence[fx.Graph],
        outer_node_dict: Dict[fx.Node, fx.Node]
    ):
        super().__init__(modules, graphs)
        
        self.outer_node_dict = outer_node_dict

    def for_each_node(self):
        from easier.core.passes.life_range_analysis import set_nodes_end_at, get_nodes_end_at

        set_nodes_end_at(self.current_node, [])
        
        return super().for_each_node()
    
    def if_call_module(self, submod: torch.nn.Module):
        if isinstance(submod, fx.GraphModule):
            ng = get_graph_module_node_group(submod)


def fuse_dataflow(modules: List[esr.Module], graphs: List[fx.Graph]):
    assert len(modules) == len(graphs) == 1
    m = modules[0]
    g = graphs[0]

    with unbalanced_compute_heartbeat(f"dataflow fusion {m.easier_hint_name}"):

        logger.info(f'Fusion pass is running for {m.easier_hint_name}')
        grouper = NodeGrouper(modules, graphs).run()
        logger.info('Fusion pass has completed')
    
    if logger.level <= logging.DEBUG:
        dump_visualized_fused_groups(modules, graphs)
    
    m, new_g = node_groups_to_graph_modules(grouper.conn_mat, m, g)

    
    return [m], [new_g]
