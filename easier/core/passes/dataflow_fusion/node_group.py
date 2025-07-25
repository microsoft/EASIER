# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List, Sequence, Set, Tuple, Union, cast, Self
from enum import Enum, auto
from typing import Optional
from torch import fx
import torch

import easier.core.module as esr
from easier.core.passes.data_dependency_analysis import \
    get_data_dependency_users
from easier.core.runtime.modules import HaloExchanger
from easier.core.runtime.metadata import \
    Role, collect_meta, get_node_meta, StructuredTensorMeta, is_node_skipped
from easier.core.passes.utils import \
    FX, OrderedSet, get_called_module, get_dag_connectivity_matrix
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
    def __init__(self, root: esr.Module, nodes: Sequence[fx.Node], id: int):
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
            self.representative_aggregators = \
                other.representative_aggregators.copy()

        self.type = other.type

        return self

    def __hash__(self) -> int:
        ret = 0
        for node in self.nodes:
            ret += hash(node)
        return ret

    def __eq__(self, other: 'NodeGroup') -> bool:
        if self.id == other.id:
            # About `assert self.nodes == other.nodes`:
            # At whatever moment, NodeGroups on the graph must be in a valid
            # state: cannot deadlock, cannot overlap, etc.
            # Although implementation-wise possible to break this rule,
            # e.g. `tmp_ng1` variables within fuse_if_no_deadlock() method,
            # such temporary states cannot go out of that method scope.
            assert self.nodes == other.nodes
            return True
        else:
            return False

    def __repr__(self) -> str:
        """
        E.g.
        ```
        NodeGroup@5(MAP, [sub_53], downstream=[[mul_33]@7, [getitem_52]@11])
        ```
        """
        type_name = self.type.name

        def _node_names(ng: NodeGroup):
            return '[' + ', '.join(node.name for node in ng.nodes) + ']'

        node_names = _node_names(self)
        downstream_names = '[' + ', '.join(
            _node_names(d_ng) + f'@{d_ng.id}'
            for d_ng in _get_downstream(self)
        ) + ']'

        return f'NodeGroup@{self.id}({type_name}, {node_names},' \
            f' downstream={downstream_names})'

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
    # TODO many checks are duplicated with NdoeGroup.infer,
    # can these methods be merged?
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
    #
    # - FX.GET_ATTR, purely syntactic Nodes
    #
    # - Aggregator Nodes whose input dist tensors are skipped
    #   (JitEngine AggregatorNeutralInputPreparation handler is nominated to
    #   run these Nodes, so we cannot put them into NodeGroups)
    #
    # - Non-aggregator replica Nodes:
    #   -   Pure replica Nodes
    #   -   nested esr.Module calls, return None -- FX.CALL_MODULE
    #   -   FX.OUTPUT, return None and are purely syntactic
    #
    # - HaloExchanger Nodes -- FX.CALL_MODULE
    meta: StructuredTensorMeta = get_node_meta(node)
    all_replica = len(
        collect_meta(meta, lambda tm: tm.role, sentinel=Role.REPLICATED)
    ) == 0

    if is_node_skipped(root, node):
        return NodeType.EXCLUDED  # skip

    if node.op == FX.GET_ATTR:
        return NodeType.EXCLUDED  # skip

    if node.target in esr.easier_aggregators:
        # TODO will esr.norm(x, p=p) have its nodes[0] be a replicated `p`?
        aggegator_input = node.all_input_nodes[0]
        if is_node_skipped(root, aggegator_input):
            return NodeType.EXCLUDED
        else:
            return NodeType.AGGREGATOR

    if all_replica:
        return NodeType.EXCLUDED

    if node.op == FX.CALL_MODULE:
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


def get_node_group_connectivity_matrix(ngs: List[NodeGroup]):
    """
    The connectivity matrix is basically a directed acyclic graph (DAG).

    Although within certain implementation like fuse_if_no_deadlock(),
    conn_mat can temporarily become cyclic, indicating deadlock occurs,
    such temporary states cannot go out of that method scope.
    """
    # TODO the distribution of NodeGroup degrees is probably ~ Poisson(1)
    # adjust shortest_path algorithm that most fits the distribution.
    return get_dag_connectivity_matrix(ngs, _get_downstream, lambda ng: ng.id)


def _get_downstream(ng: NodeGroup) -> Set[NodeGroup]:
    ret = set()
    for node in ng.nodes:
        for usr in set(
            list(node.users) + get_data_dependency_users(node)
        ).difference(ng.nodes):
            ret.add(get_node_group(usr))
    return ret


def fuse_groups_if_no_deadlock(
    ng1: NodeGroup, ng2: NodeGroup, conn_mat: torch.Tensor
) -> bool:
    """
    NodeGroups ng1 ng2 and the resultant fused NodeGroup are valid NodeGroups,
    e.g. won't have both Reducer and aggregator.

    If the fused NodeGroup has "deadlock" dataflow...

    Otherwise, fuse ng1 and ng2, merge `ng1.nodes` and `ng2.nodes`,
    and in-place modify the NodeGroup instances.
    All Nodes in the fused NodeGroup will be `set_node_group(node, ng1)`.

    I.e. NodeGroup instance ng2 will be discarded from Node metas.

    NOTE
    Because `_get_downstream()` inspects and `conn_mat` reflects
    dataflow/dependency information on the graph,
    involved NodeGroups will be modified by `ng1.copy_(fused)` temporarily.

    This will make graph-level analysis take the possibly resultant
    fused NodeGroup into consideration.

    If the analysis rejects, we reset the involved NodeGroups to the states
    when they enter this method.
    """
    assert ng1.root is ng2.root
    assert torch.all(conn_mat.diag() == 1), \
        "connectivity matrix always has diagonal"

    tmp1 = NodeGroup(ng1.root, [], ng1.id).copy_(ng1)
    tmp2 = NodeGroup(ng1.root, [], ng2.id).copy_(ng2)
    tmpmat = conn_mat.clone()

    # create a temporary conn_mat to determine if fusing the two NodeGroups
    # causes deadlock, conn_mat is still directed, but now it may have loops:
    #
    # Given NodeGroups
    #   id=P with downstream edges P->X1,X2,... and id=Q with Q->Y1,Y2,...
    #   (downstream edges are between NodeGroups)
    # After fusion we have only NodeGroup id=P and discard id=Q:
    # - forall NodeGroup U->P, add U->Y1,Y2,...    (Q discarded)
    # - forall NodeGroup V->Q, add V->X1,X2,..., V->P
    U_mask = conn_mat[:, ng1.id] > 0
    V_mask = conn_mat[:, ng2.id] > 0
    fused_row = conn_mat[ng1.id] + conn_mat[ng2.id]

    assert fused_row[ng1.id] > 0

    conn_mat[U_mask, :] += fused_row
    conn_mat[V_mask, :] += fused_row
    conn_mat.clamp_max_(1)

    # NOTE we never call `set_node_group(..., fused)` to associate
    # temporary NodeGroup instance `fused` to the graph,
    # so we cannot do checks e.g. `get_node_group(...) is not fused`.
    fused = NodeGroup(ng1.root, list(ng1.nodes) + list(ng2.nodes), ng1.id)
    ng1.copy_(fused)
    ng2.copy_(fused)

    for usr in _get_downstream(ng1):
        # If a dataflow/data-dependency user Node of the whole ng1 (fused),
        # has a downstream path back into ng1 (fused) itself, it means
        # fusing ng1 (original) and ng2 will cause a "deadlock" in the graph.
        # Then we cannot fuse ng1 and ng2.
        if conn_mat[usr.id, ng1.id] > 0:
            ng1.copy_(tmp1)
            ng2.copy_(tmp2)
            conn_mat.copy_(tmpmat)
            return False

    for node in ng1.nodes:
        set_node_group(node, ng1)

    return True


def get_node_group(node: Union[fx.Node, fx.GraphModule]) -> NodeGroup:
    if isinstance(node, fx.GraphModule):
        return getattr(node, KEY__FUSION_NODE_GROUP)
    else:
        return node.meta[KEY__FUSION_NODE_GROUP]


def set_node_group(node: Union[fx.Node, fx.GraphModule], ng: NodeGroup):
    if isinstance(node, fx.GraphModule):
        setattr(node, KEY__FUSION_NODE_GROUP, ng)
    else:
        node.meta[KEY__FUSION_NODE_GROUP] = ng
