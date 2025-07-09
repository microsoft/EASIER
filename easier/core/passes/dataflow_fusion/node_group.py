# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import itertools
from typing import List, Sequence, Set, Tuple, Union, cast
from enum import Enum, auto
from typing import Optional
from torch import fx

import easier.core.module as esr
from easier.core.passes.data_dependency_analysis import get_data_dependency_users
from easier.core.runtime.modules import HaloExchanger
from easier.core.runtime.metadata import Role, collect_meta, get_node_meta, RuntimeTensorMeta, StructuredTensorMeta, is_node_skipped
from easier.core.passes.utils import FX, OrderedSet, get_called_module
from easier.core.utils import EasierJitException

KEY__FUSION_NODE_GROUP = 'easier_fusion_nodeGroup'


class NodeType(Enum):
    EXCLUDED = auto()
    SELECT = auto()
    REDUCE = auto()
    MAP = auto()
    AGGREGATOR = auto()


class GroupType(Enum):
    EXCLUDED = auto()
    MAP = auto()
    SELECT = auto()
    REDUCE = auto()
    SELECT_REDUCE = auto()
    AGGREGATOR = auto()

class NodeGroup:
    def __init__(self, root: esr.Module, nodes: Sequence[fx.Node]) -> None:
        self.root = root
        self.nodes = OrderedSet(nodes)

        # NOTE if Reducer is skipped by JitEngine, its group is EXCLUDED
        # and won't have this field.
        self.representative_reduces: Optional[
            Tuple[esr.Reducer, List[fx.Node]]
        ] = None
        
        # Aggregators, AKA allreduces, can be grouped regardless of
        # the allreduce op (SUM PROD etc.) or the input TensorGroup.
        self.representative_aggregators: Optional[List[fx.Node]] = None

        self.type: GroupType
        self.infer_type()
    
    def __eq__(self, other: 'NodeGroup') -> bool:
        return self.nodes == other.nodes
    
    def __lt__(self, other: 'NodeGroup') -> bool:
        """
        Minimal protocol for `list.sort()`.
        """
        if _has_path(other, self):
            return False
        else:
            return True
    


    def infer_type(self) -> GroupType:
        res = GroupType.MAP
        for node in self.nodes:
            nt = get_node_type(self.root, node)

            if nt is NodeType.EXCLUDED:
                if len(self.nodes) != 1:
                    raise EasierJitException(
                        "EXCLUDED NodeGroup must be singleton"
                    )
                self.type = GroupType.EXCLUDED
                return self.type

            if nt is NodeType.REDUCE:
                reducer = cast(esr.Reducer, get_called_module(self.root, node))
                if self.representative_aggregators is not None:
                    raise EasierJitException(
                        "AGGREGATOR NodeGroup cannot have Reducers"
                    )

                if self.representative_reduces is None:
                    self.representative_reduces = (reducer, [node])
                else:
                    prev_reduer, reduce_nodes = self.representative_reduces
                    if reducer is not prev_reduer:
                        raise EasierJitException(
                            "Only one Reducer instance in a REDUCE NodeGroup"
                        )
                    reduce_nodes.append(node)
            
            if nt is NodeType.AGGREGATOR:
                if self.representative_reduces is not None:
                    raise EasierJitException(
                        "REDUCE NodeGroup cannot have aggregators"
                    )
                
                if self.representative_aggregators is None:
                    self.representative_aggregators = [node]
                else:
                    self.representative_aggregators.append(node)

            res = _infer_group_type(res, nt)
        
        self.type = res
        return res

    def all_input_nodes(self) -> OrderedSet[fx.Node]:
        """
        Get Nodes that are inputs to inner Nodes and not in this NodeGroup.
        """
        inputs = OrderedSet()
        for node in self.nodes:
            for arg in node.all_input_nodes:
                if arg not in self.nodes:
                    inputs.add(arg)
        return inputs

    def all_output_nodes(self) -> OrderedSet[fx.Node]:
        """
        Get inner Nodes which have users that are not in this NodeGroup.

        NOTE
        Multi-res operators like torch.svd() are generally grouped with
        following `getitem()` Nodes to unpack the multi-item result,
        so output Nodes of a NodeGroup won't be multi-res, i.e. in codegen
        we can treat an output Node as a single Tensor,
        without extra nested structure.
        """
        outputs = OrderedSet()
        for node in self.nodes:
            for user in node.users:
                if user not in self.nodes:
                    outputs.add(node)  # collect inner nodes
        return outputs


def _infer_group_type(gt: GroupType, nt: NodeType) -> GroupType:
    # TODO many checks are duplicated with NdoeGroup.infer, can these methods be merged?
    assert nt is not NodeType.EXCLUDED, \
        "EXCLUDED Nodes are never grouped together"

    if gt is GroupType.MAP:
        if nt is NodeType.SELECT:
            return GroupType.SELECT
        elif nt is NodeType.REDUCE:
            return GroupType.REDUCE
        elif nt is NodeType.AGGREGATOR:
            return GroupType.AGGREGATOR
        else:
            assert nt in [NodeType.MAP]
            return GroupType.MAP
    
    elif gt is GroupType.SELECT:
        if nt is NodeType.REDUCE:
            return GroupType.SELECT_REDUCE
        elif nt is NodeType.AGGREGATOR:
            return GroupType.AGGREGATOR
        else:
            assert nt in [NodeType.MAP, NodeType.SELECT]
            return GroupType.SELECT
    
    elif gt in [GroupType.REDUCE, GroupType.SELECT_REDUCE]:
        if nt is NodeType.SELECT:
            return GroupType.SELECT_REDUCE
        elif nt is NodeType.AGGREGATOR:
            raise EasierJitException("REDUCE and AGGREGATOR cannot be grouped")
        else:
            assert nt in [NodeType.MAP, NodeType.REDUCE]
            return gt
    
    elif gt is GroupType.AGGREGATOR:
        if nt is NodeType.REDUCE:
            raise EasierJitException("REDUCE and AGGREGATOR cannot be grouped")
        else:
            assert nt in [NodeType.MAP, NodeType.SELECT, NodeType.AGGREGATOR]
            return GroupType.AGGREGATOR
    
    else:
        raise EasierJitException(f"Unknown GroupType {gt}")



def get_node_type(root: esr.Module, node: fx.Node) -> NodeType:
    # Skip:
    # - JitEngine skipped Nodes, distributed but batchsize==0;
    # - FX.GET_ATTR, purely syntactic Nodes
    # - Non-aggregator replica Nodes:
    #   -   Pure replica Nodes
    #   -   nested esr.Module calls, return None -- FX.CALL_MODULE
    #   -   FX.OUTPUT, return None and are purely syntactic
    # - HaloExchanger Nodes -- FX.CALL_MODULE
    meta: StructuredTensorMeta = get_node_meta(node)
    all_replica = len(
        collect_meta(meta, lambda tm: tm.role, sentinel=Role.REPLICATED)
    ) == 0

    if is_node_skipped(node):
        return NodeType.EXCLUDED  # skip

    if node.op == FX.GET_ATTR:
        return NodeType.EXCLUDED  # skip

    if all_replica:
        if node.target not in esr.easier_aggregators:
            return NodeType.EXCLUDED  # skip
        else:
            return NodeType.AGGREGATOR
    
    if node.op is FX.CALL_MODULE:
        submod = get_called_module(root, node)

        if isinstance(submod, HaloExchanger):
            return NodeType.EXCLUDED  # skip
        elif isinstance(submod, esr.Reducer):
            return NodeType.REDUCE
        elif isinstance(submod, esr.Selector):
            return NodeType.SELECT
        else:
            raise EasierJitException(f'Unsupported CALL_MODULE {submod}')
    
    return NodeType.MAP

def _has_path(n1: NodeGroup|fx.Node, n2: NodeGroup) -> bool:
        # n1: Union[NodeGroup, fx.Node], n2: Union[NodeGroup, fx.Node]) -> bool:
    if n1 == n2:
        return True
    else:
        for usr in _get_downstream(n1):
            if _has_path(usr, n2):
                return True
        return False

def _get_downstream(
        input: Union[NodeGroup, fx.Node]) -> Set[Union[NodeGroup, fx.Node]]:
    ret = set()
    if isinstance(input, fx.Node):
        for usr in set(list(input.users) + get_data_dependency_users(input)):
            if get_node_group(usr):
                ret.add(get_node_group(usr))
            else:
                ret.add(usr)

    else:
        for node in input.nodes:
            for usr in set(list(node.users) + get_data_dependency_users(node)):
                if usr not in input.nodes:
                    if get_node_group(usr):
                        ret.add(get_node_group(usr))
                    else:
                        ret.add(usr)
    return ret


def try_fuse_node_groups(ng1: NodeGroup, ng2: NodeGroup) -> bool:
    """
    NodeGroups ng1 ng2 and the resultant fused NodeGroup are valid NodeGroups,
    e.g. won't have both Reducer and aggregator.

    But if the fused NodeGroup has "deadlock" dataflow
    """
    assert ng1.root is ng2.root
    fused = NodeGroup(ng1.root, list(ng1.nodes) + list(ng2.nodes))

    for usr in _get_downstream(fused):
        if _has_path(usr, fused):
            return False
    
    for node in fused.nodes:
        set_node_group(node, fused)
    return True


def get_node_group(node: fx.Node) -> Optional[NodeGroup]:
    return node.meta.get(KEY__FUSION_NODE_GROUP, None)


def set_node_group(node: fx.Node, ng: NodeGroup):
    node.meta[KEY__FUSION_NODE_GROUP] = ng
