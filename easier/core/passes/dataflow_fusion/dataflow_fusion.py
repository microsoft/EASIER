# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List, Sequence
from torch import fx
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

    def _create_node_group(self, node: fx.Node) -> NodeGroup:
        ng = NodeGroup(self.current_module, [node])
        set_node_group(node, ng)

        logger.debug(node)

        for user in node.users:
            downstream_ng = get_node_group(user)
            if downstream_ng is None:
                downstream_ng = self._create_node_group(user)

            if ng is not downstream_ng and self._check_rules(ng, downstream_ng) == FuseDecision.PASS:
                try_fuse_node_groups(ng, downstream_ng)

        return ng
    
    def for_each_node(self):
        if get_node_group(self.current_node) is None:
            self._create_node_group(self.current_node)


def fuse_dataflow(modules: List[esr.Module], graphs: List[Graph]):
    with unbalanced_compute_heartbeat("dataflow fusion"):

        # import cProfile
        # from easier.core.runtime.dist_env import get_default_dist_env
        # rank = get_default_dist_env().rank
        # p = cProfile.Profile()
        # p.enable()

        logger.info('Fusion pass is running')
        NodeGrouper(modules, graphs).run()
        logger.info('Fusion pass has completed')

        # p.disable()
        # p.dump_stats(f'fusion_{rank}.prof')

        m, = modules
        g, = graphs
        ngs = set()
        for n in g.nodes:
            ng = get_node_group(n)
            ngs.add(ng)

        logger.info(m.__class__.__name__)
        logger.info(g)
        for ng in ngs:
            if ng.type != GroupType.EXCLUDED:
                logger.info(ng)

    return modules, graphs