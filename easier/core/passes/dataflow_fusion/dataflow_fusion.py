# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import networkx
from typing import Dict, List, Sequence, Set
import scipy
from torch import fx
import torch
from torch.fx.graph import Graph

import easier.core.module as esr
from easier.core.passes.dataflow_fusion.node_group import \
    NodeGroup, get_node_group, set_node_group, fuse_groups_if_no_deadlock, \
    get_node_group_connectivity_matrix
from easier.core.passes.dataflow_fusion.debug import \
    dump_visualized_fused_groups
from easier.core.passes.dataflow_fusion.fusion_rules import \
    GroupRule, NodeRule, FuseDecision
import easier.core.passes.dataflow_fusion.fusion_rules as rules
from easier.core.passes.utils import EasierInterpreter, SubmodNameAllocator
from easier.core.utils import logger
from easier.core.runtime.dist_env import unbalanced_compute_heartbeat


class NodeGrouper(EasierInterpreter):
    def __init__(self, modules: Sequence[esr.Module], graphs: Sequence[Graph]):
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
    ng_conn_mat: torch.Tensor, module: esr.Module, graph: Graph
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
    ngs: List[NodeGroup] = list(set(map(get_node_group, graph.nodes)))
    ngids = [ng.id for ng in ngs]
    assert len(set(ngids)) == len(ngids)

    ids_tensor = torch.tensor(ngids)
    active_ng_conn_mat = ng_conn_mat[ids_tensor, ids_tensor]

    dag = networkx.DiGraph()
    dag.add_nodes_from(range(len(ngids)))

    for rowid, src_ngid in enumerate(ngids):
        for colid in active_ng_conn_mat[rowid].argwhere().ravel().tolist():
            dag.add_edge(rowid, colid)
    
    topo_rowids: List[int] = list(networkx.topological_sort(dag))

    raw_node_ids: Dict[fx.Node, int] = dict(
        (n, i) for i, n in enumerate(graph.nodes)
    )

    fused_g = Graph()
    gm_name_allocator = SubmodNameAllocator('easier_kernel')
    node_env: Dict[fx.Node, fx.Node] = {}

    for rowid in topo_rowids:
        ng = ngs[rowid]

        if len(ng.nodes) > 1:

            gm_node_env: Dict[fx.Node, fx.Node] = {}
            for in_node in ng.all_input_nodes():


            ng_raw_nodes = sorted(ng.nodes, key=raw_node_ids.__getitem__)

            # e.g. easier_kernel3_select_reduce
            gm_name = gm_name_allocator.alloc_name(
                module, f"_{ng.type.name.lower()}"
            )
            module.add_module(gm_name, )

            for raw_n in ng.nodes:
                node_env[raw_n] = 1
        else:
            raw_n, = ng.nodes
            new_n = fused_g.node_copy(raw_n, node_env.__getitem__)
            node_env[raw_n] = new_n

    return module, fused_g

def fuse_dataflow(modules: List[esr.Module], graphs: List[Graph]):
    assert len(modules) == len(graphs) == 1
    m = modules[0]
    g = graphs[0]

    with unbalanced_compute_heartbeat(f"dataflow fusion {m.easier_hint_name}"):

        logger.info(f'Fusion pass is running for {m.easier_hint_name}')
        grouper = NodeGrouper(modules, graphs).run()
        logger.info('Fusion pass has completed')
    
    if logger.level <= logging.DEBUG:
        dump_visualized_fused_groups(modules, graphs)
    
    m, g = node_groups_to_graph_modules(grouper.conn_mat, m, g)
    return [m], [g]
