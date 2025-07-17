# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List, Sequence, Set
from torch import fx
import torch
from torch.fx.graph import Graph

import easier.core.module as esr
from easier.core.passes.dataflow_fusion.node_group import GroupType, NodeGroup, get_node_group, set_node_group, try_fuse_node_groups
from easier.core.passes.dataflow_fusion.fusion_rules import GroupRule, NodeRule, FuseDecision
import easier.core.passes.dataflow_fusion.fusion_rules as rules
from easier.core.passes.utils import EasierInterpreter
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

        root, = modules
        graph, = graphs

        # Init NodeGroup for each Node, as NodeGroup.ctor will calculate
        # downstream NodeGroups, we need to traverse in the reversed list.
        nodes = list(graph.nodes)
        for id, n in reversed(list(enumerate(nodes))):
            ng = NodeGroup(root, [n], id)
            set_node_group(n, ng)
        
        self._init_connectivity(nodes)
    
    def _init_connectivity(self, nodes: List[fx.Node]):
        num = len(nodes)

        conn_mat = torch.full((num, num), fill_value=0, dtype=torch.int64)
        for node in nodes:
            ng = get_node_group(node)
            for downstream_ng in ng.downstream:
                conn_mat[ng.id][downstream_ng.id] = 1
            
            for upstream_ng in ng.get_upstream():
                conn_mat[upstream_ng.id][ng.id] = 1
        
        logger.info(conn_mat)
        assert torch.equal(conn_mat, conn_mat.T)



    def _check_rules(self, ng1: NodeGroup, ng2: NodeGroup) -> FuseDecision:
        """
        Check if any rule rejects, in a blacklist manner.

        ng1 is graph-wise upstream to ng2.

        Returns:
        -   REJECT if any rules reject, PASS if all rules pass.
        """
        logger.debug(f"Checking {ng1} {ng2}")
        for group_rule in self.group_rules:
            if group_rule.check(ng1, ng2) == FuseDecision.REJECT:
                logger.debug(f"{group_rule.__class__.__name__} reject")
                return FuseDecision.REJECT

        for n1 in ng1.nodes:
            for n2 in ng2.nodes:
                if n2 in n1.users:
                    for node_rule in self.node_rules:
                        if node_rule.check(n1, n2) == FuseDecision.REJECT:
                            logger.debug(f"{node_rule.__class__.__name__} reject")
                            return FuseDecision.REJECT

        logger.debug('pass')
        return FuseDecision.PASS

    def _merge_downstream_groups(self, node: fx.Node, _rec_depth=0) -> NodeGroup:
        self.visited.add(node)

        ng = get_node_group(node)

        logger.debug(f'{_rec_depth} -> {node}')

        for user in node.users:
            if user not in self.visited:
                downstream_ng = self._merge_downstream_groups(user, _rec_depth+1)
            else:
                downstream_ng = get_node_group(user)

            if ng is not downstream_ng and self._check_rules(ng, downstream_ng) == FuseDecision.PASS:
                try_fuse_node_groups(ng, downstream_ng)

                # If fusion succeeds:
                # - NodeGroup instance `downstream_ng` is discarded;
                # - NodeGroup instance `ng` is in-place expanded;
                # - All Nodes in the fused group are bound to
                #   NodeGroup instance `ng`.
        
        logger.debug(f'{_rec_depth} END')

        return ng
    
    def for_each_node(self):
        if self.current_node not in self.visited:
            self._merge_downstream_groups(self.current_node)

def fuse_dataflow(modules: List[esr.Module], graphs: List[Graph]):
    with unbalanced_compute_heartbeat("dataflow fusion"):

        # import cProfile
        # from easier.core.runtime.dist_env import get_default_dist_env
        # rank = get_default_dist_env().rank
        # p = cProfile.Profile()
        # p.enable()

        m, = modules
        g, = graphs
        logger.info(m.__class__.__name__)
        logger.info(g)

        logger.info('Fusion pass is running')
        NodeGrouper(modules, graphs).run()
        logger.info('Fusion pass has completed')

        # p.disable()
        # p.dump_stats(f'fusion_{rank}.prof')

        ngs = set()
        for n in g.nodes:
            ng = get_node_group(n)
            ngs.add(ng)

        for ng in ngs:
            if ng.type != GroupType.EXCLUDED:
                logger.info(ng)

    return modules, graphs