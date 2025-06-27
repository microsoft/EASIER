# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, List, Sequence, Tuple, Union
from dataclasses import dataclass
from typing_extensions import Literal

import torch
from torch.fx.graph import Graph
from torch.fx.node import Node

import numpy as np
import scipy.sparse
from torch.nn.modules import Module

import easier.core.module as esr
from easier.core.passes.tensor_grouping import \
    EasierTensorGroup, get_node_tensor_group, get_tensor_groups_relation
from easier.core.passes.utils import \
    EasierInterpreter, OrderedSet, \
    get_selector_reducer_idx_partition_pair, \
    normalize_reducer_call_into_args, normalize_selector_call_into_args, \
    get_easier_tensors
from easier.core.runtime.dist_env import \
    get_runtime_dist_env, unbalanced_compute_heartbeat
from easier.core.distpart import distpart_kway, DistConfig
from easier.core.utils import EasierJitException, logger

# METIS adj weights are ints
METIS_ADJWGT_REDUCER: int = 10


def parallel_partition_graph(
    world_size: int, rank: int,
    subadjmat_height: int, adjmat_width: int, vtxdist: torch.Tensor,
    rowcolids_and_cps: List[Tuple[torch.Tensor, torch.Tensor, 'CommPair']]
):
    """
    Args:
    -   vtxdist:
            Specifies how the `indptr` is splitted on workers,
            its values are the offsets to `indptr` vector, e.g. [0, N, 2N, ...]
            where N stands for the length of a slice in `indptr`,
            i.e. the adjmat_height.
    -   rowcolids_and_causes:
            Row IDs are local row indexes to the sub adj mat,
            i.e. the upperbound is subadjmat_height ~= adjmat_width/world_size.

            Both global row IDs and col IDs (col IDs are always global)
            must form a symmetric sparse matrix.
            (No matter how previous subpasses remap the IDs)
    """
    # NOTE
    # - `csr_matrix` sums up duplicated matrix cells during construction
    # - `csr_matrix` automatically choose int32/int64 for its `indptr` and
    #   `indices` ndarrays regarding upperbounds of the height and the weight.
    with unbalanced_compute_heartbeat("assemble sparse adjmat"):

        selector_graph = scipy.sparse.csr_matrix(
            (subadjmat_height, adjmat_width), dtype=np.int32)
        reducer_graph = scipy.sparse.csr_matrix(
            (subadjmat_height, adjmat_width), dtype=np.int32)

        # Edge weights in the adjmat are:
        # - if involved in Selectors, no matter how many times, weights are 1
        # - each time involved in Reducers, those weights are increased by 10,
        #   no matter how many edges there are.
        for (commpair_rowids, commpair_colids, comm_pair) in rowcolids_and_cps:
            assert commpair_rowids.ndim == commpair_colids.ndim == 1
            assert commpair_rowids.shape == commpair_rowids.shape

            commpair_rowids = commpair_rowids.detach().cpu().numpy()
            commpair_colids = commpair_colids.detach().cpu().numpy()

            csr_ones = np.ones((commpair_rowids.shape[0],), dtype=np.int32)
            commpair_graph = scipy.sparse.csr_matrix(
                (csr_ones, (commpair_rowids, commpair_colids)),  # sum up dups
                shape=(subadjmat_height, adjmat_width)
            )

            if comm_pair.caused_by_reducer:
                # Clamp potentially summed up elements,
                # so that this adjmat for some reducer, as a whole,
                # has a single non-1 weight value.
                commpair_graph = commpair_graph.minimum(1)
                commpair_graph = commpair_graph * METIS_ADJWGT_REDUCER

                # It's efficient to add if both are in CSR format.
                reducer_graph = reducer_graph + commpair_graph

            else:
                selector_graph = selector_graph + commpair_graph

            # Before processing there is always a collective call, so we can
            # just record the processed event.
            logger.debug(f"Sub adjmat for {comm_pair} processed")

        graph = selector_graph.minimum(1) + reducer_graph

    local_membership = distpart_kway(
        DistConfig(
            int(vtxdist[-1]),
            (vtxdist[1:] - vtxdist[:-1]).tolist()
        ),
        torch.from_numpy(graph.indptr).to(torch.int64),
        torch.from_numpy(graph.indices).to(torch.int64),
        torch.from_numpy(graph.data).to(torch.int64)
    )

    return local_membership


@dataclass
class CommPair:
    src_tensor_group: EasierTensorGroup
    src_idx_partition: torch.Tensor  # may be any integer dtype

    dst_tensor_group: EasierTensorGroup
    dst_idx_partition: torch.Tensor

    caused_by_reducer: bool
    cause_hint: str = ""

    def get_symmetric_pair(self):
        return CommPair(
            src_tensor_group=self.dst_tensor_group,
            src_idx_partition=self.dst_idx_partition,
            dst_tensor_group=self.src_tensor_group,
            dst_idx_partition=self.src_idx_partition,

            # The original source, communication direction is irrelevant.
            caused_by_reducer=self.caused_by_reducer,
            cause_hint=self.cause_hint
        )

    def __repr__(self) -> str:
        src = self.src_tensor_group
        dst = self.dst_tensor_group
        return f"CommPair({src.hint} -{self.cause_hint}-> {dst.hint})"


class CommPairCollector(EasierInterpreter):
    def __init__(
        self,
        modules: Sequence[esr.Module], graphs: Sequence[Graph],
    ):
        super().__init__(modules, graphs)

        self.visited = set()

        # Only Select-ed and Reduce-ed TensorGroups are involved in these:
        # (i.e. TensorGroups only referenced by "get_attr" are not included)
        self.tensor_groups: OrderedSet[EasierTensorGroup] = OrderedSet()

        # Does not include the symmetric part for the communication operations.
        self.comm_pairs: List[CommPair] = []

    def if_call_module(self, submod: Module):
        if isinstance(submod, esr.Module):  # nested esr.Module calls
            return

        if submod in self.visited:
            # We assume the TensorGroup equivalency based on module instance
            return

        node = self.current_node

        if isinstance(submod, esr.Selector):
            input_node = normalize_selector_call_into_args(
                *node.args, **node.kwargs)

            caused_by_reducer = False

        elif isinstance(submod, esr.Reducer):
            input_node, _out_node = normalize_reducer_call_into_args(
                *node.args, **node.kwargs)

            caused_by_reducer = True

        else:
            raise EasierJitException(
                f'{type(submod)} is not supported to appear in'
                ' tensor grouping'
            )

        self.visited.add(submod)
        assert isinstance(input_node, Node)
        # For Reducer, even `out=` parameter is given, the Node itself
        # has metadata as output too.

        src_tensor_group = get_node_tensor_group(input_node)
        dst_tensor_group = get_node_tensor_group(node)
        assert src_tensor_group is not None
        assert dst_tensor_group is not None

        self.tensor_groups.add(src_tensor_group)
        self.tensor_groups.add(dst_tensor_group)

        src_idx, dst_idx = get_selector_reducer_idx_partition_pair(submod)
        self.comm_pairs.append(CommPair(
            src_tensor_group, src_idx,
            dst_tensor_group, dst_idx,
            caused_by_reducer,
            submod.easier_hint_name
        ))


def _calculate_group_offsets(tensor_groups: OrderedSet[EasierTensorGroup]):
    dist_env = get_runtime_dist_env()
    world_size = dist_env.world_size

    zero_leading_grp_sizes = torch.tensor(
        [0] + [g.n for g in tensor_groups], dtype=torch.int64
    )
    grp_cluster_size_beforelast = int(
        (zero_leading_grp_sizes // world_size).sum()
    )

    grps_offsets_in_clusters_beforelast: Dict[EasierTensorGroup, int] = dict(
        zip(
            tensor_groups,
            torch.cumsum(  # the first 0 keeps unchanged
                zero_leading_grp_sizes // world_size, dim=0
            )[:-1].tolist()
        )
    )
    grps_offsets_in_cluster_lastrank: Dict[EasierTensorGroup, int] = dict(
        zip(
            tensor_groups,
            torch.cumsum(  # the first 0 keeps unchanged
                zero_leading_grp_sizes - (
                    zero_leading_grp_sizes // world_size * (world_size - 1)
                ),  # cumsum with accumulated remainders
                dim=0
            )[:-1].tolist()
        )
    )

    return zero_leading_grp_sizes, grp_cluster_size_beforelast, \
        grps_offsets_in_clusters_beforelast, grps_offsets_in_cluster_lastrank


def partition_tensor_groups_with_adjmat(
    tensor_groups: OrderedSet[EasierTensorGroup],
    comm_pairs: List[CommPair]
):
    """
    Args:
    -   tensor_groups:
            In the order of how related Selector/Reducer appears in the IR.
    -   comm_pairs:
            Communication patterns appearing in the program.
            For each operation of communication (e.g. Reducer.forward)
            only the original one (i.e. the direction) CommPair instance
            should be passed in, this function will add its symmetric part
            to the adjmat.

    Intuitively, we need to concat all elements from all TensorGroups:
    into a list, and construct adjmat between elements on this list.

    E.g. world_size==3
        TensorGroup A has elements [A0, ..., A10]
        TensorGroup B has elements [B0, ..., B15]
        TensorGroup C has elements [C0, ..., C20]

    Instead of directly concat into [A0,...,A10, B0,...,B15, C0,...,C20]
    we first split each TensorGroup into `world_size` segments:
        TensorGroup A becomes { [A0, ..., A2], [A3, ..., A5], [A6, ..., A10] }
        ...
    The mapping is `A_n` is put into the `min(n//world_size, world_size-1)`-th
    segment, and div remainders are simply put on the last segment.

    Then, segments for the same rank are concat-ed to form a _cluster_:
        [A0,...,A2, B0,...,A4, C0,...,C6]  # length = 3 + 5 + 7

    and the last cluster will have different lengths because of remainders:
        [A6,...,A10, B10,...,B15, C14,...,C20]  # length = 5 + 6 + 7

    The final list of the 3 clusters will be:
        [
            A0,...,A2,  B0,...,A4,   C0,...,C6,    # subadjmat for rank-0
            A3,...,A5,  B5,...,A9,   C7,...,C13,   # subadjmat for rank-1
            A6,...,A10, B10,...,B15, C14,...,C20,  # subadjmat for rank-2
        ]
    and the adjmat between list elements can naturally be divided into
    `world_size` subadjmats.
    (note, both rows and columns of the adjmat must be reordered,
    as the adjmat is symmetric.)

    Such split and reorganization would balance non-zero entities among
    the workers.
    """
    # TODO tensor_group_partition didn't handle the cases well
    # where the number of vertexes is less than the world size.
    # (although such lightweight cases do not need distribution at all)

    comm_pairs = [
        pair
        for unidirect_pair in comm_pairs
        for pair in [unidirect_pair, unidirect_pair.get_symmetric_pair()]
    ]

    dist_env = get_runtime_dist_env()
    world_size = dist_env.world_size
    rank = dist_env.rank

    #
    # Construct local sub adjmat
    # by finding all row and col numbers according to communication
    #

    # TODO currently we collect all data then exchange them between workers,
    # therefore there is expected to be a memory peak. We may need to
    # send and free data in stream to decrease the memory peak.

    zero_leading_grp_sizes, grp_cluster_size_beforelast, \
        grps_offsets_in_clusters_beforelast, \
        grps_offsets_in_cluster_lastrank = _calculate_group_offsets(
            tensor_groups
        )

    # rowids are subadjmat-local, i.e. with upperbound (sum(n)//world_size).
    # [(concat_rowids, concat_colids, caused_by_reducer)]
    rowcolids_for_commpairs: List[Tuple[
        torch.Tensor, torch.Tensor, CommPair
    ]] = []

    def _consider_not_partitioning_if_too_small(grp: EasierTensorGroup):
        perworker_n = grp.n // world_size
        if grp.n == 0 or perworker_n == 0:
            raise EasierJitException(
                f"Consider not partitioning {grp.hint} with only"
                f" {grp.n} elements"
            )

    for comm_pair in comm_pairs:
        row_tensor_group = comm_pair.src_tensor_group
        rowgrp_idx_part = comm_pair.src_idx_partition
        col_tensor_group = comm_pair.dst_tensor_group
        colgrp_idx_part = comm_pair.dst_idx_partition

        # xxxgrp_per_worker_n can be 0, where (a small set of) all elements
        # are put on the last worker.
        # P.S. Even group.n can be 0.
        rowgrp_per_worker_n = row_tensor_group.n // world_size
        colgrp_per_worker_n = col_tensor_group.n // world_size

        _consider_not_partitioning_if_too_small(row_tensor_group)
        _consider_not_partitioning_if_too_small(col_tensor_group)

        # zip(rowgrp_ids_w, colgrp_ids_w) are for worker-w
        # - rowgrp IDs are subadjmat-local row IDs
        # - colgrp IDs are adjmat col IDs
        rowgrp_ids_to_send: List[torch.Tensor] = []
        colgrp_ids_to_send: List[torch.Tensor] = []

        for w in range(dist_env.world_size):
            rowgrp_idx_lb_w = rowgrp_per_worker_n * w
            if w == world_size - 1:
                rowgrp_offset_in_cluster: int = \
                    grps_offsets_in_cluster_lastrank[row_tensor_group]

                rowgrp_idx_ub_w: int = row_tensor_group.n
            else:
                rowgrp_offset_in_cluster: int = \
                    grps_offsets_in_clusters_beforelast[row_tensor_group]

                rowgrp_idx_ub_w: int = rowgrp_per_worker_n * (w + 1)

            idx_part_pos_w = torch.logical_and(
                rowgrp_idx_lb_w <= rowgrp_idx_part,
                rowgrp_idx_part < rowgrp_idx_ub_w
            )

            #
            # map (split only) row idxes into adjmat col IDs
            #
            rowgrp_idx_part_w = rowgrp_idx_part[idx_part_pos_w]
            rowgrp_ids_w = \
                rowgrp_idx_part_w - rowgrp_idx_lb_w + rowgrp_offset_in_cluster

            #
            # map (split + scatter segments) col idxes into adjmat col IDs
            #
            colgrp_idx_part_w = colgrp_idx_part[idx_part_pos_w]

            colgrp_idx_part_w_cluster_ids = (
                colgrp_idx_part_w // colgrp_per_worker_n
                # last cluster has more than colgrp_perw_n
            ).clamp(max=world_size - 1)

            # remainders are in [0, colgrp_per_worker_n) if before last, or
            # [0, colgrp.n - colgrp_per_worker_n * (world_size-1))
            colgrp_idx_part_w_cluster_remainders = \
                colgrp_idx_part_w - \
                colgrp_idx_part_w_cluster_ids * colgrp_per_worker_n

            idx_is_before_last_rank = \
                colgrp_idx_part_w_cluster_ids < (world_size - 1)

            colgrp_ids_w = torch.full_like(colgrp_idx_part_w, fill_value=-1)

            def _for_column_clustersbefore_or_lastcluster(
                clusters_kind_mask: torch.Tensor,
                colgrps_offsets_in_cluster: Dict[EasierTensorGroup, int]
            ) -> None:
                # We have two kinds of clusters:
                # - clusters before the last;
                # - the last cluster itself.
                colgrp_offset_in_cluster: int = colgrps_offsets_in_cluster[
                    col_tensor_group
                ]

                clusters_offsets = \
                    colgrp_idx_part_w_cluster_ids[clusters_kind_mask] \
                    * grp_cluster_size_beforelast

                remainders_in_cluster = \
                    colgrp_idx_part_w_cluster_remainders[clusters_kind_mask]

                colgrp_ids_kind = \
                    clusters_offsets + colgrp_offset_in_cluster \
                    + remainders_in_cluster

                assert torch.all(colgrp_ids_w[clusters_kind_mask] == -1), \
                    "cluster ids should not be set twice"
                colgrp_ids_w[clusters_kind_mask] = colgrp_ids_kind

            # for all ranks before the last: equal offsets and cluster sizes
            _for_column_clustersbefore_or_lastcluster(
                idx_is_before_last_rank, grps_offsets_in_clusters_beforelast
            )
            _for_column_clustersbefore_or_lastcluster(
                ~idx_is_before_last_rank, grps_offsets_in_cluster_lastrank
            )

            assert torch.all(colgrp_ids_w != -1), "cluster ids all assigned"

            rowgrp_ids_to_send.append(rowgrp_ids_w.to(dist_env.comm_device))
            colgrp_ids_to_send.append(colgrp_ids_w.to(dist_env.comm_device))

        # Both kinds of ids are 0-based, relative to the row/col TensorGroups
        # need to be added with Groups' offset before being filled into adjmat.
        rowgrp_ids = [
            t.cpu() for t in dist_env.all_to_all(rowgrp_ids_to_send)
        ]
        colgrp_ids = [
            t.cpu() for t in dist_env.all_to_all(colgrp_ids_to_send)
        ]

        # rowgrp IDs are subadjmat-local row IDs
        # colgrp IDs are adjmat col IDs
        rowcolids_for_commpairs.append((
            torch.concat(rowgrp_ids),
            torch.concat(colgrp_ids),
            comm_pair
        ))
    # endfor comm_pairs

    #
    # Invoke partition
    #

    # Each worker owns a length-(gro.n/world_size) part,
    # all parts are considered to be concat-ed to form a subadjmat.
    vtxdist = torch.arange(world_size + 1) * int(
        (zero_leading_grp_sizes // world_size).sum()
    )
    vtxdist[-1] = zero_leading_grp_sizes.sum()

    subadjmat_height = int(vtxdist[rank + 1] - vtxdist[rank])

    local_membership = parallel_partition_graph(
        world_size, rank,
        subadjmat_height=subadjmat_height, adjmat_width=int(vtxdist[-1]),
        vtxdist=vtxdist,
        rowcolids_and_cps=rowcolids_for_commpairs)

    return local_membership


@dataclass
class ElemPartArangeIdx:
    """
    Stands for an "orphan" TensorGroup that's not involved in Selector/Reducer,
    even not in the `easier.Module.forward()` scope to `easier.compile()`.
    Such orphan TensorGroups still need a partition.
    Evenly partitioning would suffice.

    This dataclass serves as a descriptor for such simple cases, and will
    save data transfer and serialization.
    """
    # basically, arguments to torch.arange
    start: int
    end: int


@dataclass
class ElemPartReorderedArangeIdx:
    """
    A shuffled ArangeIdx. Generally resulted from ElemPart reordering on
    ArangeIdx ElemPart.
    """
    start: int
    end: int


@dataclass
class ElemPartSortedIdx:
    """
    This idx_desc will appear when any general ElemPart is made, or, determined
    to be sorted.
    Some inspection algorithm which used to use e.g. torch.isin()
    could benefit from the sorted-ness.

    NOTE a temp sorted ElemPart could be created to locally boost analysis,
    even the sorted ElemPart will not get stored.
    """
    pass


"""
4 idx_desc types, including None, form a partial order of "well-ordered-ness"

       ___  ReorderedArangeIdx ___
None --|                         |-- ArangeIdx
       ---  SortedIdx  -----------
"""


@dataclass
class ElemPart:

    # Only for this worker.
    idx_desc: Union[
        None, ElemPartArangeIdx, ElemPartReorderedArangeIdx, ElemPartSortedIdx
    ]

    idx: torch.Tensor

    # All lengths are replicated on all workers
    lengths: List[int]

    hint: str

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return id(self) == id(other)


def synchronize_partition_result(
    tensor_groups: OrderedSet[EasierTensorGroup],
    local_membership: torch.Tensor
) -> Dict[EasierTensorGroup, ElemPart]:
    """
    Synchronize partition results ("elempart") into each TensorGroup.

    Remark:
    Synchronization is needed because partition results of a single
    TensorGroup may be scattered on multiple workers.
    And the synchronized partition data is only about the TensorGroup partition
    specific to current worker.
    """
    #
    # Exchange partition result
    # to construct partition info of every TensorGroup on every worker
    #
    dist_env = get_runtime_dist_env()
    world_size = dist_env.world_size
    rank = dist_env.rank

    zero_leading_grp_sizes, grp_cluster_size_beforelast, \
        _grps_offsets_in_clusters_beforelast, \
        _grps_offsets_in_cluster_lastrank = _calculate_group_offsets(
            tensor_groups
        )

    if rank == world_size - 1:
        grps_offsets = _grps_offsets_in_cluster_lastrank
    else:
        grps_offsets = _grps_offsets_in_clusters_beforelast

    synced_elemparts: Dict[EasierTensorGroup, ElemPart] = {}

    # local_membership is for "interleavedly concated" subadjmat whose rows are
    # [perw_nrow_sum * rank, max(perw_nrow_sum * (rank + 1), accum_n)

    for grp_i, tensor_group in enumerate(tensor_groups):
        grp_perworker_n = tensor_group.n // world_size
        grp_last_n = tensor_group.n - grp_perworker_n * (world_size - 1)

        if rank == world_size - 1:
            grp_this_n = grp_last_n
        else:
            grp_this_n = grp_perworker_n

        grp_local_membership_begin = grps_offsets[tensor_group]
        grp_local_membership_end = grp_local_membership_begin + grp_this_n

        grp_membership = local_membership[
            grp_local_membership_begin:grp_local_membership_end
        ]
        grp_epidx_offset = grp_perworker_n * rank

        elempart_idxes_by_rank = []

        for w in range(world_size):
            elempart_idx_w = \
                grp_epidx_offset + torch.argwhere(grp_membership == w).ravel()
            elempart_idxes_by_rank.append(
                elempart_idx_w.to(dist_env.comm_device)
            )

        elempart_idxes = [
            t.cpu() for t in dist_env.all_to_all(elempart_idxes_by_rank)
        ]
        elempart_idx = torch.concat(elempart_idxes)

        elempart_lengths = dist_env.all_gather_into_tensor(
            torch.tensor([elempart_idx.shape[0]], device=dist_env.comm_device)
        ).tolist()

        elempart_i = len(synced_elemparts)
        elempart_hint = get_elempart_hint(elempart_i, tensor_group)

        # TODO regardless how to partition TensorGroup and then concat segments
        # into adjmat, the "element identities" in the Group are still
        # incrementally ordered, so the resultant ElemPart can be made
        # SortedIdx initially.
        synced_elemparts[tensor_group] = ElemPart(
            None, elempart_idx, elempart_lengths, hint=elempart_hint
        )

    return synced_elemparts


# NOTE the two kinds of hints may have the same "elempart_i" subtext,
# but the postfix ("arange" or empty) can still differentiate ElemParts.


def get_elempart_hint(elempart_i: int, tensor_group: EasierTensorGroup):
    return f'{elempart_i}:{tensor_group.hint}'


def get_arange_elempart_hint(elempart_i: int, tensor_group: EasierTensorGroup):
    return f'{elempart_i}:{tensor_group.hint}:arange'


class CommTensorGroupGetter(EasierInterpreter):
    def __init__(self, modules, graphs) -> None:
        super().__init__(modules, graphs)

        self.tensor_groups: OrderedSet[EasierTensorGroup] = OrderedSet()

    def if_call_module(self, submod: Module):
        if isinstance(submod, (esr.Selector, esr.Reducer)):
            src_tensor_group, dst_tensor_group = get_tensor_groups_relation(
                self.current_node, submod
            )
            self.tensor_groups.add(src_tensor_group)
            self.tensor_groups.add(dst_tensor_group)


def get_even_elemparts(modules, graphs):
    """
    An even ElemPart stands for an evenly distributed partition
    taking no communication structure into consideration at all.

    Such an ElemPart/partition is usually assigned to an EasierTensorGroup
    that's not involved in communication at all;
    Or when the global partition mode is specified as 'evenly'.

    This method will:
    1.  additional to `comm_tensor_groups`,
        traverse all esr.Tensor instances and collect their EasierTensorGroups
        in a consistent order, too;
    2.  for any TensorGroup that are not assigned with a better
        ElemPart/partition, i.e. not in `comm_elemparts: dict`,
        assign an even ElemPart for it.
    """
    dist_env = get_runtime_dist_env()
    world_size = dist_env.world_size
    rank = dist_env.rank

    group_getter = CommTensorGroupGetter(modules, graphs).run()
    tensor_groups: OrderedSet[EasierTensorGroup] = group_getter.tensor_groups

    # Some esr.Tensors are not related to esr.Selector/Reducer,
    # or may not even be referenced in esr.Modules at all,
    # we must ensure all esr.Tensors are assigned an (at least even) ElemPart,
    # because these esr.Tensors may still get `.collect()` or `.save()`
    # where we need valid elemparts to reconstruct their full data.
    named_dtensor: Dict[esr.Tensor, List[Tuple[int, str]]] = \
        get_easier_tensors(modules)
    for p, roots_attrs in named_dtensor.items():
        if not p.is_partition:
            continue
        tensor_groups.add(p.easier_tensor_group)

    elemparts: Dict[EasierTensorGroup, ElemPart] = {}

    for elempart_i, tensor_group in enumerate(tensor_groups):
        n = tensor_group.n

        per_worker_n = n // world_size
        residue = n % world_size

        # given the quantity of residue below:
        # assert 0 <= residue < world_size
        # we simply let the last worker have all these residual elements

        start = per_worker_n * rank
        end = start + per_worker_n
        if rank + 1 == world_size:
            end = n

        lengths = [
            per_worker_n
            for w in range(world_size)
        ]
        lengths[-1] += residue

        elempart_hint = get_arange_elempart_hint(elempart_i, tensor_group)

        elemparts[tensor_group] = ElemPart(
            ElemPartArangeIdx(start, end),
            torch.arange(start, end),
            lengths,
            hint=elempart_hint
        )

    return elemparts


def partition_tensor_groups(modules: List[esr.Module], graphs: List[Graph]):
    # TODO handle when len is 0 -- needs a global quality refinement later
    modes = set(mod.partition_mode for mod in modules)
    assert len(modes) == 1
    partition_mode: Literal['metis', 'evenly'] = modes.pop()  # type: ignore

    world_size = get_runtime_dist_env().world_size

    elemparts = get_even_elemparts(modules, graphs)

    # Overwrite with better partitions for some TensorGroups
    if partition_mode == 'metis' and world_size > 1:
        comm_pairs_collector = CommPairCollector(modules, graphs)
        comm_pairs_collector.run()

        # Communication-related ElemParts
        # (e.g. TensorGroups for esr.Tensors not involved in communication
        # are not included)
        comm_elemparts: Dict[EasierTensorGroup, ElemPart]
        if len(comm_pairs_collector.comm_pairs) > 0:
            # always check comm_pairs > 0 to exclude the real cases where
            # there is really no communication.
            local_membership = partition_tensor_groups_with_adjmat(
                comm_pairs_collector.tensor_groups,
                comm_pairs_collector.comm_pairs)

            comm_elemparts = synchronize_partition_result(
                comm_pairs_collector.tensor_groups,
                local_membership
            )
            assert set(comm_elemparts.keys()).issubset(elemparts.keys())

            elemparts.update(comm_elemparts)

    for root in modules:
        root.easier_elemparts = elemparts

    return modules, graphs
