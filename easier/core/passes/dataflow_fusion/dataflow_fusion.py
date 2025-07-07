# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List, Sequence
from torch import fx
from torch.fx.graph import Graph

import easier.core.module as esr
from easier.core.passes.dataflow_fusion.node_group import NodeGroup, get_node_group, set_node_group, fuse_node_groups
from easier.core.passes.dataflow_fusion.fusion_rules import GroupRule, NodeRule, RuleDecision
from easier.core.passes.utils import FX, EasierInterpreter, get_called_module
from easier.core.runtime.modules import HaloExchanger
from easier.core.runtime.metadata import get_node_meta, RuntimeTensorMeta, StructuredTensorMeta, is_node_skipped, is_node_replica


class NodeGrouper(EasierInterpreter):
    def __init__(self, modules: Sequence[esr.Module], graphs: Sequence[Graph]):
        assert len(modules) == len(graphs) == 1
        super().__init__(modules, graphs)

        self.group_rules: List[GroupRule] = []
        self.node_rules: List[NodeRule] = []

    def _check_rules(self, ng1: NodeGroup, ng2: NodeGroup) -> RuleDecision:
        """
        Check if any rule rejects, in a blacklist manner.

        Returns:
        -   REJECT if any rules reject, PASS if all rules pass.
        """
        for rule in self.group_rules:
            if rule.check(ng1, ng2) == RuleDecision.REJECT:
                return RuleDecision.REJECT

        for n1 in ng1.nodes:
            for n2 in ng2.nodes:
                if n2 in n1.users:
                    # TODO how about n1 in n2.users?
                    for rule in self.node_rules:
                        if rule.check(n1, n2) == RuleDecision.REJECT:
                            return RuleDecision.REJECT

        return RuleDecision.PASS

    def _create_node_group(self, node: fx.Node) -> NodeGroup:
        ng = NodeGroup(self.current_module, [node])
        set_node_group(node, ng)

        for user in node.users:
            downstream_ng = get_node_group(user)
            if downstream_ng is None:
                downstream_ng = self._create_node_group(user)

            if ng is not downstream_ng and self._check_rules(ng, downstream_ng) == RuleDecision.PASS:
                fuse_node_groups(ng, downstream_ng)

        return ng

    def for_each_node(self):
        # Skip:
        # - JitEngine skipped Nodes, distributed but batchsize==0;
        # - FX.GET_ATTR, purely syntactic Nodes
        # - Non-aggregator replica Nodes:
        #   -   Pure replica Nodes
        #   -   nested esr.Module calls, return None
        #   -   FX.OUTPUT, return None and are purely syntactic
        # - HaloExchanger Nodes
        if is_node_skipped(self.current_node) \
            or self.current_node.op == FX.GET_ATTR \
            or (
            is_node_replica(self.current_node) and
            not self.current_node.target in esr.easier_aggregators
        ) \
            or (
            self.current_node.op == FX.CALL_MODULE and
            isinstance(
                get_called_module(self.current_module, self.current_node),
                HaloExchanger
            )
        ):
            # set_node_group(self.current_node, "TRIVIAL_GROUP")
            return None

        return super().for_each_node()
