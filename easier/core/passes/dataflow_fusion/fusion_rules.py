# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod
from enum import Enum, auto
from torch import fx

from easier.core.passes.dataflow_fusion.node_group import GroupType, NodeGroup


class RuleDecision(Enum):
    REJECT = auto()
    PASS = auto()


class GroupRule(ABC):
    """
    Generally symmetric.
    """
    @abstractmethod
    def check(self, ng1: NodeGroup, ng2: NodeGroup) -> RuleDecision:
        pass


class RejectExcludedGroup(GroupRule):
    def check(self, ng1: NodeGroup, ng2: NodeGroup) -> RuleDecision:
        if ng1.type() == GroupType.EXCLUDED or ng2.type() == GroupType.EXCLUDED:
            return RuleDecision.REJECT
        else:
            return RuleDecision.PASS


class FuseSameReducerGroups(GroupRule):
    def check(self, ng1: NodeGroup, ng2: NodeGroup) -> RuleDecision:
        pass


class ReducerAggregatorGroupsConflict(GroupRule):
    def check(self, ng1: NodeGroup, ng2: NodeGroup) -> RuleDecision:
        ng1_type = ng1.type()
        ng2_type = ng2.type()
        if (
            ng1_type in [GroupType.REDUCE, GroupType.SELECT_REDUCE] and
            ng2_type is GroupType.AGGREGATOR
        ) or (
            ng2_type in [GroupType.REDUCE, GroupType.SELECT_REDUCE] and
            ng1_type is GroupType.AGGREGATOR
        ):
            return RuleDecision.REJECT
        else:
            return RuleDecision.PASS


class _FuseReduceGroupAndOtherGroup(GroupRule):
    pass


class NodeRule(ABC):
    """
    Dataflow arg and user.
    """
    @abstractmethod
    def check(self, arg: fx.Node, user: fx.Node) -> RuleDecision:
        pass

class GroupingEndsAtAggregator(NodeRule):
    def check(self, arg: fx.Node, user: fx.Node) -> RuleDecision:
        if "arg is aggregator":
            return RuleDecision.REJECT
        else:
            return RuleDecision.PASS