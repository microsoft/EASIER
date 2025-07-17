# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import itertools
from typing import List, Sequence, Set, Tuple, Union, cast, Self
from enum import Enum, auto
from typing import Optional
from torch import fx

import easier.core.module as esr
from easier.core.passes.data_dependency_analysis import get_data_dependency_inputs, get_data_dependency_users
from easier.core.runtime.dist_env import get_runtime_dist_env
from easier.core.runtime.modules import HaloExchanger
from easier.core.runtime.metadata import Role, collect_meta, get_node_meta, RuntimeTensorMeta, StructuredTensorMeta, is_node_skipped
from easier.core.passes.utils import FX, OrderedSet, get_called_module
from easier.core.utils import EasierJitException, logger

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
    def __init__(self, root: esr.Module, nodes: Sequence[fx.Node], id: int) -> None:
        self.root = root
        self.nodes = OrderedSet(nodes)
        self.id = id

        # NOTE if Reducer is skipped by JitEngine, its group is EXCLUDED
        # and won't have this field.
        self.representative_reduces: Optional[
            Tuple[esr.Reducer, List[fx.Node]]
        ] = None
        
        # Aggregators, AKA allreduces, can be grouped regardless of
        # the allreduce op (SUM PROD etc.) or the input TensorGroup.
        self.representative_aggregators: Optional[List[fx.Node]] = None

        self.type: GroupType
        self._infer_type()

        self.downstream: Set[NodeGroup]
        self.reset_downstream()

    def copy_(self, other: 'NodeGroup') -> Self:
        self.root = other.root
        self.nodes = OrderedSet(other.nodes)
        self.id = other.id

        self.representative_reduces = None
        if other.representative_reduces is not None:
            (reducer, reduces) = other.representative_reduces
            self.representative_reduces = (reducer, reduces.copy())

        self.representative_aggregators = None
        if other.representative_aggregators is not None:
            self.representative_aggregators = other.representative_aggregators.copy()

        self.type = other.type

        self.downstream = set(other.downstream)

        return self
    
    def __hash__(self) -> int:
        ret = 0
        for node in self.nodes:
            ret += hash(node)
        return ret

    def __eq__(self, other: 'NodeGroup') -> bool:
        return self.nodes == other.nodes
    
    def __lt__(self, other: 'NodeGroup') -> bool:
        if _has_path(other, self):
            return False
        else:
            return True
    
    def __repr__(self) -> str:
        """
        E.g.
        ```
        NodeGroup(MAP, [sub_53], downstream=[[mul_33], [getitem_52]])
        ```
        """
        type_name = self.type.name

        def _node_names(ng: NodeGroup):
            return '[' + ', '.join(node.name for node in ng.nodes) + ']'

        node_names = _node_names(self)
        downstream_names = '[' + ', '.join(_node_names(d_ng) for d_ng in self.downstream) + ']'

        return f'NodeGroup(type={type_name}, nodes={node_names}, downstream={downstream_names})'
    


    def _infer_type(self) -> GroupType:
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
    
    def reset_downstream(self) -> Set['NodeGroup']:
        # TODO cut down on GT.EXCLUDED
        downstream = set()
        for node in self.nodes:
            for usr in set(list(node.users) + get_data_dependency_users(node)).difference(self.nodes):
                # if usr not in self.nodes:
                usr_ng = get_node_group(usr)
                downstream.add(usr_ng)
        self.downstream = downstream
        return downstream
    
    def get_upstream(self) -> Set['NodeGroup']:
        # TODO cut down on GT.EXCLUDED
        upstream = set()
        for node in self.nodes:
            for input in set(list(node.all_input_nodes) + get_data_dependency_inputs(node)).difference(self.nodes):
                # if input not in self.nodes:
                input_ng = get_node_group(input)
                upstream.add(input_ng)
        return upstream

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

def _has_path(n1: NodeGroup, n2: NodeGroup) -> bool:
    """
    Detect if there is a dataflow/data-dependency path from n1 to n2.
    """
        # n1: Union[NodeGroup, fx.Node], n2: Union[NodeGroup, fx.Node]) -> bool:
    # if isinstance(n1, fx.Node):
    #     n1_nodes = OrderedSet([n1])
    # else:
    #     n1_nodes = n1.nodes

    # TODO will we meet that [n1:Node] == n2.nodes ? 

    if get_runtime_dist_env().rank == 0:
        logger.debug(f'? has path {n1} {n2}')

    # if n1_nodes == n2.nodes:
    if n1 == n2:
        return True
    else:
        for usr in n1.downstream:
            if _has_path(usr, n2):
                return True
        return False

# def _get_downstream(
#         input: Union[NodeGroup, fx.Node]) -> Set[Union[NodeGroup, fx.Node]]:
#     ret = set()
#     if isinstance(input, fx.Node):
#         for usr in set(list(input.users) + get_data_dependency_users(input)):
#             if get_node_group(usr):
#                 ret.add(get_node_group(usr))
#             else:
#                 ret.add(usr)

#     else:
#         for node in input.nodes:
#             for usr in set(list(node.users) + get_data_dependency_users(node)):
#                 if usr not in input.nodes:
#                     if get_node_group(usr):
#                         ret.add(get_node_group(usr))
#                     else:
#                         ret.add(usr)
#     return ret


def try_fuse_node_groups(ng1: NodeGroup, ng2: NodeGroup) -> bool:
    """
    NodeGroups ng1 ng2 and the resultant fused NodeGroup are valid NodeGroups,
    e.g. won't have both Reducer and aggregator.

    If the fused NodeGroup has "deadlock" dataflow...

    Otherwise, fuse ng1 and ng2, merge `ng1.nodes` and `ng2.nodes`,
    and in-place modify the NodeGroup instances.
    All Nodes in the fused NodeGroup will be `set_node_group(node, ng1)`.

    I.e. NodeGroup instance ng2 will be discarded from Node metas.

    NOTE
    Because _get_downstream() and _has_path() methods above
    inspect dataflow/dependency information on the graph,
    involved NodeGroups will be modified by `ng1.copy_(fused)` temporarily.

    This will make graph-level analysis take the possibly resultant
    fused NodeGroup into consideration.

    If the analysis rejects, we reset the involved NodeGroups to the states
    when they enter this method.
    """
    assert ng1.root is ng2.root
    tmp1 = NodeGroup(ng1.root, [], ng1.id).copy_(ng1)
    tmp2 = NodeGroup(ng1.root, [], ng2.id).copy_(ng2)

    # NOTE we never call `set_node_group(..., fused)` to associate
    # temporary NodeGroup instance `fused` to the graph,
    # so we cannot do checks e.g. `get_node_group(...) is not fused`.
    fused = NodeGroup(ng1.root, list(ng1.nodes) + list(ng2.nodes), ng1.id)
    ng1.copy_(fused)
    ng2.copy_(fused)

    for usr in ng1.downstream:
        # If a dataflow/data-dependency user Node of the whole ng1 (fused),
        # has a downstream path back into ng1 (fused) itself, it means
        # fusing ng1 (original) and ng2 will cause a "deadlock" in the graph.
        # Then we cannot fuse ng1 and ng2.
        if _has_path(usr, ng1):
            ng1.copy_(tmp1)
            ng2.copy_(tmp2)
            return False

    for node in ng1.nodes:
        set_node_group(node, ng1)

    for up_ng in ng1.get_upstream():
        up_ng.reset_downstream()
    
    return True

def sort_node_groups():
    "TODO"

def get_node_group(node: fx.Node) -> NodeGroup:
    return node.meta[KEY__FUSION_NODE_GROUP]


def set_node_group(node: fx.Node, ng: NodeGroup):
    node.meta[KEY__FUSION_NODE_GROUP] = ng
