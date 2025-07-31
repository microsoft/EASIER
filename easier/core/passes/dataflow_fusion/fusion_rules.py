# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod
from enum import Enum, auto
from torch import fx

from easier.core.passes.dataflow_fusion.node_group import \
    GroupType, NodeGroup, NodeType, get_node_group, get_node_type
from easier.core.passes.utils import normalize_reducer_call_into_args


class FuseDecision(Enum):
    REJECT = auto()
    PASS = auto()


class NodeRule(ABC):
    """
    Dataflow arg and user.
    """
    @abstractmethod
    def check(self, arg: fx.Node, user: fx.Node) -> FuseDecision:
        pass


class GroupingEndsBeforeSelector(NodeRule):
    def check(self, arg: fx.Node, user: fx.Node) -> FuseDecision:
        user_ng = get_node_group(user)
        assert user_ng is not None

        if get_node_type(user_ng.root, user) == NodeType.SELECT:
            return FuseDecision.REJECT
        else:
            return FuseDecision.PASS


class GroupingEndsAfterAggregator(NodeRule):
    def check(self, arg: fx.Node, user: fx.Node) -> FuseDecision:
        arg_ng = get_node_group(arg)
        assert arg_ng is not None

        if get_node_type(arg_ng.root, arg) == NodeType.AGGREGATOR:
            return FuseDecision.REJECT
        else:
            return FuseDecision.PASS


"""
The scheduling for hardware threads of Reducer are generally inconsistent
with the scheduling of Mapped ops on Reducer output tensors.

Reducer outputs include:
-   Reducer's dataflow users
    -   rejected by GroupingEndsAfterReducer

-   Reducer's in-place out: `forward(input, out=self.x)`
    -   rejected by GroupingEndsAtReducerOut
"""


class GroupingEndsAfterReducer(NodeRule):
    # TODO this rule and EndsAfterAggregator rule can be seen as
    # GroupRule (or Group-Node rule), which can be checked faster by
    # rep_reducer and other_ng.nodes,
    # rather than being checked at O(#self.nodes * #other.nodes) complexity.
    def check(self, arg: fx.Node, user: fx.Node) -> FuseDecision:
        arg_ng = get_node_group(arg)
        assert arg_ng is not None

        if get_node_type(arg_ng.root, arg) == NodeType.REDUCE:
            return FuseDecision.REJECT
        else:
            return FuseDecision.PASS


class GroupingEndsAtReducerOut(NodeRule):
    def check(self, arg: fx.Node, user: fx.Node) -> FuseDecision:
        user_ng = get_node_group(user)
        assert user_ng is not None

        if get_node_type(user_ng.root, user) == NodeType.REDUCE:
            r_in, r_out = normalize_reducer_call_into_args(
                *user.args, **user.kwargs
            )
            if arg is r_out:
                return FuseDecision.REJECT

        return FuseDecision.PASS


class GroupRule(ABC):
    """
    Generally symmetric on both argument NodeGroups.
    """
    @abstractmethod
    def check(self, ng1: NodeGroup, ng2: NodeGroup) -> FuseDecision:
        pass


class RejectExcludedGroup(GroupRule):
    """
    GroupRule runs first, then this will cover the rejection of EXCLUDED Nodes.

    Reject:
    -   JitEngine skipped Nodes
    -   GET_ATTR, purely syntactic
    -   HaloExchanger
    -   non-aggregator replica, e.g. pure replica and esr.Module calls.
    """

    def check(self, ng1: NodeGroup, ng2: NodeGroup) -> FuseDecision:
        if ng1.type == GroupType.EXCLUDED or ng2.type == GroupType.EXCLUDED:
            return FuseDecision.REJECT
        else:
            return FuseDecision.PASS


class ReducerAggregatorGroupsConflict(GroupRule):
    def check(self, ng1: NodeGroup, ng2: NodeGroup) -> FuseDecision:
        if self._reject(ng1, ng2) or self._reject(ng2, ng1):
            return FuseDecision.REJECT
        else:
            return FuseDecision.PASS

    def _reject(self, ng1: NodeGroup, ng2: NodeGroup) -> bool:
        return \
            ng1.representative_aggregators is not None \
            and ng2.representative_reduces is not None


# NOTE unlike Reducers, different aggregators (esr.sum, esr.max, etc.)
# won't conflict, they can be grouped as long as their input Tensors
# are of the same TensorGroup.


class ReducerInstancesConflict(GroupRule):
    def check(self, ng1: NodeGroup, ng2: NodeGroup) -> FuseDecision:
        if ng1.representative_reduces is not None \
                and ng2.representative_reduces is not None:
            r1 = ng1.representative_reduces[0]
            r2 = ng2.representative_reduces[0]
            if r1 is not r2:
                return FuseDecision.REJECT

        return FuseDecision.PASS
