# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union, cast
from dataclasses import dataclass
from typing_extensions import Literal, OrderedDict, TypeAlias
import functools
import more_itertools
import os

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
    get_cpu_dist_env
from easier.core.utils import EasierJitException, logger

# METIS adj weights are ints
METIS_ADJWGT_REDUCER: int = 10


def parallel_partition_graph(
    world_size: int, rank: int,
    subadjmat_height: int, adjmat_width: int, vtxdist: torch.Tensor,
    rowcolids_and_causes: List[Tuple[torch.Tensor, torch.Tensor, bool]]
):
    """
    Args:
    -   vtxdist:
            Specifies how the `indptr` is splitted on workers,
            its values are the offsets to `indptr` vector, e.g. [0, N, 2N, ...]
            where N stands for the length of a slice in `indptr`,
            i.e. the adjmat_height.
    """

    # NOTE
    # - `csr_matrix` sums up duplicated matrix cells during construction
    # - `csr_matrix` automatically choose int32/int64 for its `indptr` and
    #   `indices` ndarrays regarding upperbounds of the height and the weight.
    # - `parmetis.part_kway` can call 32bit/64bit underlying native library
    #   depending on its `xadj` argument i.e. `graph.indptr`.
    #   Other arguments, including weights, will be casted to that int type.
    selector_graph = scipy.sparse.csr_matrix(
        (subadjmat_height, adjmat_width), dtype=np.int32)
    reducer_graph = scipy.sparse.csr_matrix(
        (subadjmat_height, adjmat_width), dtype=np.int32)

    # Edge weights in the adjmat are:
    # - if involved in Selectors, no matter how many times, weights are 1
    # - each time involved in Reducers, those weights are increased by 10,
    #   no matter how many edges there are.
    for (commpair_rowids, commpair_colids, by_reducer) in rowcolids_and_causes:
        assert commpair_rowids.ndim == commpair_colids.ndim == 1
        assert commpair_rowids.shape == commpair_rowids.shape

        commpair_rowids = commpair_rowids.detach().cpu().numpy()
        commpair_colids = commpair_colids.detach().cpu().numpy()

        csr_ones = np.ones((commpair_rowids.shape[0],), dtype=np.int32)
        commpair_graph = scipy.sparse.csr_matrix(
            (csr_ones, (commpair_rowids, commpair_colids)),  # sum up dups
            shape=(subadjmat_height, adjmat_width)
        )

        if by_reducer:
            # Clamp potentially summed up elements,
            # so that this adjmat for some reducer, as a whole,
            # has a single non-1 weight value.
            commpair_graph = commpair_graph.minimum(1)
            commpair_graph = commpair_graph * METIS_ADJWGT_REDUCER

            # It's efficient to add if both are in CSR format.
            reducer_graph = reducer_graph + commpair_graph

        else:
            selector_graph = selector_graph + commpair_graph

    graph = (selector_graph.minimum(1) + reducer_graph).tolil()
    # scipy warns against `setdiag` on CSR. LIL format is recommended instead.

    # ParMETIS require the diagonal (relatively to the global adjmat)
    # are zeros and excluded from sparsity.
    off_diag: int = int(vtxdist[rank])
    graph.setdiag(0, off_diag)
    graph = graph.tocsr()

    metis_impl = os.environ.get('EASIER_METIS_IMPL', 'EASIER').upper()
    if metis_impl == 'EASIER':
        from easier.core.distpart import distpart_kway, DistConfig
        local_membership = distpart_kway(
            DistConfig(
                int(vtxdist[-1]),
                (vtxdist[1:] - vtxdist[:-1]).tolist()
            ),
            torch.from_numpy(graph.indptr).to(torch.int64),
            torch.from_numpy(graph.indices).to(torch.int64),
            torch.from_numpy(graph.data).to(torch.int64)
        )

    elif metis_impl == 'PARMETIS':
        from mgmetis import parmetis
        from mpi4py import MPI
        import time
        comm: MPI.Intracomm = MPI.COMM_WORLD
        parmetis_start = time.time()
        # `ncuts` is already summed up and replicated;
        # `local_membership` works like this:
        #   the result of`AllGather(local_membership)` is the result of
        #   non-distributed version of graph partitioning.
        ncuts, local_membership = parmetis.part_kway(
            comm.size, graph.indptr, graph.indices, vtxdist, comm,
            adjwgt=graph.data)

        parmetis_latency = time.time() - parmetis_start
        logger.debug(
            f"ParMetis finished: ncuts={ncuts},"
            f" total time: {parmetis_latency}sec"
        )

        local_membership = torch.tensor(local_membership)

    else:
        raise EasierJitException('Unknown Metis implementation')

    return local_membership


@dataclass
class CommPair:
    src_tensor_group: EasierTensorGroup
    src_idx_partition: torch.Tensor  # may be any integer dtype

    dst_tensor_group: EasierTensorGroup
    dst_idx_partition: torch.Tensor

    caused_by_reducer: bool

    def get_symmetric_pair(self):
        return CommPair(
            src_tensor_group=self.dst_tensor_group,
            src_idx_partition=self.dst_idx_partition,
            dst_tensor_group=self.src_tensor_group,
            dst_idx_partition=self.src_idx_partition,

            # The original source, communication direction is irrelevant.
            caused_by_reducer=self.caused_by_reducer
        )


class CommPairCollector(EasierInterpreter):
    def __init__(
        self,
        modules: Sequence[esr.Module], graphs: Sequence[Graph],
    ):
        super().__init__(modules, graphs)

        self.visited = set()

        # Only Select-ed and Reduce-ed TensorGroups are involved in these:
        # (i.e. TensorGroups only referenced by "get_attr" are not included)
        self.tensor_group_to_matedge_offsets: \
            OrderedDict[EasierTensorGroup, int] = OrderedDict()
        self.accum_n = 0

        # Does not include the symmetric part for the communication operations.
        self.comm_pairs: List[CommPair] = []

    def _init_tensor_group_offset(self, tensor_group: EasierTensorGroup):
        if tensor_group not in self.tensor_group_to_matedge_offsets:
            offset = self.accum_n
            self.tensor_group_to_matedge_offsets[tensor_group] = offset
            self.accum_n += tensor_group.n

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

        self._init_tensor_group_offset(src_tensor_group)
        self._init_tensor_group_offset(dst_tensor_group)

        src_idx, dst_idx = get_selector_reducer_idx_partition_pair(submod)
        self.comm_pairs.append(CommPair(src_tensor_group, src_idx,
                                        dst_tensor_group, dst_idx,
                                        caused_by_reducer))


def partition_tensor_groups_with_adjmat(
    tensor_group_to_matedge_offsets: OrderedDict[EasierTensorGroup, int],
    accum_n: int,
    comm_pairs: List[CommPair]
):
    """
    Args:
    -   tensor_group_to_matedge_offsets:
            In the order of how related Selector/Reducer appears in the IR.
    -   comm_pairs:
            Communication patterns appearing in the program.
            For each operation of communication (e.g. Reducer.forward)
            only the original one (i.e. the direction) CommPair instance
            should be passed in, this function will add its symmetric part
            to the adjmat.

    This worker is responsible to fill in adjmat entities in
    rows within `[grp_row_lb, grp_row_ub)`. But the indexes specifying adjmat
    entities may locate in other workers.

    ```
                  |         |
        x x x x-----------D----------    --SubAdjMatSpanLB[y]=0
        x     x E                    )
        x     x   |     F   |        ) worker0
        x x x x     G                )
        | e     y y y y y y y y   B  )   --GroupOffset[y]=4  --GrpRowLB[y,0]=0
    - - | -  -  y  -  -  -  - y -  - ) - --GrpRowLB[y,1]=1   ~~GrpRowUB[y,0]=1
        |     g y |         | y      )
        |       y             y      ) worker1
        |   f   y |         | y     C)
        d       y             y A    )
    - - | -  -  y |-  -  -  | y -  - ) - ~~GrpRowUB[y,1]=6
        |       y y y y y y y y      )
        |                 a     z z z) worker2  ~~GroupOffset[y]+N[y]=12
        |       b |         |   z   z)
        |               c       z z z)
         ~~~~~~~~~|~~~~~~~~~|~~~~~~~~    ~~SubAdjMatSpanUB[y]=3
    ```
    The horizontal and vertical lines in the figure stand for both
    the upper bounds of the previous regions (i.e. exclusive) as well as
    the lower bounds of the next regions (i.e. inclusive).

    For example, `Select([5,0,4], Y)` from Group-y into Group-z will fill in
    matrix entities pointed by `A B C`, across two workers;
    and symmetrically, entities `a b c` will also be filled in.

    While `Reduce([5,0,4,2], X)` from Group-x into Group-y will fill in entities
    `D E F G` and symmetrically `d e f g`.
    """

    comm_pairs = [
        pair
        for unidirect_pair in comm_pairs
        for pair in [unidirect_pair, unidirect_pair.get_symmetric_pair()]
    ]

    dist_env = get_cpu_dist_env()
    world_size = dist_env.world_size
    rank = dist_env.rank
    per_worker_n = (accum_n + world_size - 1) // world_size
    empty_buffer = torch.empty((0,), dtype=torch.int64,
                               device=dist_env.comm_device)

    #
    # Construct local sub adjmat
    # by finding all row and col numbers according to communication
    #

    # TODO currently we collect all data then exchange them between workers,
    # therefore there is expected to be a memory peak. We may need to
    # send and free data in stream to decrease the memory peak.

    # [(concat_rowidx, concat_colidx, caused_by_reducer)]
    rowcolids_for_commpairs: List[Tuple[torch.Tensor, torch.Tensor, bool]] = []

    for comm_pair in comm_pairs:
        row_tensor_group = comm_pair.src_tensor_group
        rowgrp_idx = comm_pair.src_idx_partition
        col_tensor_group = comm_pair.dst_tensor_group
        colgrp_idx = comm_pair.dst_idx_partition

        rowgrp_offset = tensor_group_to_matedge_offsets[row_tensor_group]
        next_rowgrp_offset = rowgrp_offset + row_tensor_group.n
        colgrp_offset = tensor_group_to_matedge_offsets[col_tensor_group]

        # Limit the region to search
        sub_adjmat_span_lb = rowgrp_offset // per_worker_n
        sub_adjmat_span_ub = (next_rowgrp_offset - 1) // per_worker_n + 1

        subadjmat_rowids_to_send: List[torch.Tensor] = \
            [empty_buffer] * world_size
        adjmat_colids_to_send: List[torch.Tensor] = [empty_buffer] * world_size

        for w in range(sub_adjmat_span_lb, sub_adjmat_span_ub):
            # Region of rows of adjmat on worker-w
            rowgrp_rowid_lb = max(per_worker_n * w, rowgrp_offset)
            rowgrp_rowid_ub = min(per_worker_n * (w + 1), next_rowgrp_offset)

            # The range of indexes for that region on worker-w
            rowgrp_idx_lb: int = rowgrp_rowid_lb - rowgrp_offset
            rowgrp_idx_ub: int = rowgrp_rowid_ub - rowgrp_offset

            # `idx_pos.dtype == torch.bool`: this is bool-slicing
            idx_pos = torch.logical_and(rowgrp_idx_lb <= rowgrp_idx,
                                        rowgrp_idx < rowgrp_idx_ub)
            rowgrp_idx_slice = rowgrp_idx[idx_pos]
            colgrp_idx_slice = colgrp_idx[idx_pos]

            # Adjust row_grp_idx (of domain `[0,row_group.n)`)
            # into row ids of sub adjmat on worker-w
            subadjmat_rowids = (rowgrp_idx_slice +
                                rowgrp_offset) % per_worker_n
            # But `colgrp_idx` does not need adjustment since adjmat is not
            # partitioned among columns.
            adjmat_colids = colgrp_idx_slice + colgrp_offset

            subadjmat_rowids_to_send[w] = subadjmat_rowids
            adjmat_colids_to_send[w] = adjmat_colids

        subadjmat_rowids_tensors = \
            dist_env.all_to_all(subadjmat_rowids_to_send)
        adjmat_colids_tensors = \
            dist_env.all_to_all(adjmat_colids_to_send)

        # concat all recieved row/col ids
        # and the result is not ordered or uniqued.
        rowcolids_for_commpairs.append((
            torch.concat(subadjmat_rowids_tensors),
            torch.concat(adjmat_colids_tensors),
            comm_pair.caused_by_reducer
        ))
    # endfor comm_pairs

    #
    # Invoke ParMETIS
    #

    # e.g. [0, N, 2N, ..., min(accum_n, wN)]
    vtxdist = torch.arange(world_size + 1) * per_worker_n
    vtxdist[-1].clamp_(max=accum_n)

    subadjmat_height = int(vtxdist[rank + 1] - vtxdist[rank])

    local_membership = parallel_partition_graph(
        world_size, rank,
        subadjmat_height=subadjmat_height, adjmat_width=accum_n,
        vtxdist=vtxdist,
        rowcolids_and_causes=rowcolids_for_commpairs)

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
class ElemPart:

    # Only for this worker.
    idx_desc: Union[torch.Tensor, ElemPartArangeIdx]

    # All lengths are replicated on all workers
    lengths: List[int]

    hint: str

    @functools.cached_property
    def idx(self) -> torch.Tensor:
        if isinstance(self.idx_desc, ElemPartArangeIdx):
            return torch.arange(self.idx_desc.start, self.idx_desc.end)
        else:
            return self.idx_desc

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return id(self) == id(other)


def synchronize_partition_result(
    tensor_group_to_matedge_offsets: OrderedDict[EasierTensorGroup, int],
    accum_n: int,
    local_membership: torch.Tensor
) -> Dict[EasierTensorGroup, ElemPart]:
    """
    Synchronize ParMETIS-partition results ("elempart") into each TensorGroup.

    Remark:
    Synchronization is needed because partition results of a single
    TensorGroup may be scattered on multiple workers.
    And the synchronized partition data is only about the TensorGroup partition
    specific to current worker.
    """
    #
    # Exchange ParMETIS result
    # to construct partition info of every TensorGroup on every worker
    #
    dist_env = get_cpu_dist_env()
    world_size = dist_env.world_size
    rank = dist_env.rank

    empty_buffer = torch.empty((0,), dtype=torch.int64,
                               device=dist_env.comm_device)

    per_worker_n = (accum_n + world_size - 1) // world_size
    subadjmat_rowid_lb = per_worker_n * rank
    subadjmat_rowid_ub = min(per_worker_n * (rank + 1), accum_n)

    synced_elemparts: Dict[EasierTensorGroup, ElemPart] = {}

    # `local_membership` is the partition result for this local sub adjmat,
    # which represents rows within `[rowid_lb, rowid_ub)` of adjmat
    # and might (partially) cover multiple TensorGroups.
    # We need to communicate and gather TensorGroup elements assigned to
    # this worker.
    for tensor_group, offset in tensor_group_to_matedge_offsets.items():
        elempart_to_send: List[torch.Tensor] = [empty_buffer] * world_size
        tensor_group_batch_size = tensor_group.n

        if offset >= subadjmat_rowid_ub \
                or offset + tensor_group_batch_size < subadjmat_rowid_lb:
            # This local sub adjmat doesn't cover this TensorGroup.
            pass  # do nothing

        else:
            # `tensor_group` is (partially) covered by `local_membership`
            # and the overlapped part of `local_membership` is:
            local_membership_begin = \
                max(offset, subadjmat_rowid_lb) - subadjmat_rowid_lb
            local_membership_end = \
                min(offset + tensor_group_batch_size, subadjmat_rowid_ub) \
                - subadjmat_rowid_lb

            # .. and this overlapped part specifies membership
            # for TensorGroup elements within:
            tensor_group_begin = max(0, subadjmat_rowid_lb - offset)
            tensor_group_end = min(tensor_group_batch_size,
                                   subadjmat_rowid_ub - offset)

            assert local_membership_end - local_membership_begin \
                == tensor_group_end - tensor_group_begin

            grp_membership = local_membership[
                local_membership_begin:local_membership_end]

            for w in range(world_size):
                elempart = tensor_group_begin \
                    + torch.argwhere(grp_membership == w).ravel()
                elempart_to_send[w] = elempart

        elempart_recv = dist_env.all_to_all(elempart_to_send)

        elempart = torch.concat(elempart_recv)
        elempart_lengths = dist_env.all_gather_into_tensor(
            torch.tensor([elempart.shape[0]], device=dist_env.comm_device)
        ).tolist()

        assert torch.equal(elempart.unique(sorted=True), elempart.sort()[0])

        elempart_i = len(synced_elemparts)
        elempart_hint = get_elempart_hint(elempart_i, tensor_group)

        synced_elemparts[tensor_group] = ElemPart(
            elempart, elempart_lengths, hint=elempart_hint
        )

    # endfor tensor_groups

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
    dist_env = get_cpu_dist_env()
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
            lengths,
            hint=elempart_hint
        )

    return elemparts


def partition_tensor_groups(modules: List[esr.Module], graphs: List[Graph]):
    # TODO handle when len is 0 -- needs a global quality refinement later
    modes = set(mod.partition_mode for mod in modules)
    assert len(modes) == 1
    partition_mode: Literal['metis', 'evenly'] = modes.pop()  # type: ignore

    elemparts = get_even_elemparts(modules, graphs)

    # Overwrite with better partitions for some TensorGroups
    if partition_mode == 'metis':
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
                comm_pairs_collector.tensor_group_to_matedge_offsets,
                comm_pairs_collector.accum_n,
                comm_pairs_collector.comm_pairs)

            comm_elemparts = synchronize_partition_result(
                comm_pairs_collector.tensor_group_to_matedge_offsets,
                comm_pairs_collector.accum_n,
                local_membership
            )
            assert set(comm_elemparts.keys()).issubset(elemparts.keys())

            elemparts.update(comm_elemparts)

    for root in modules:
        root.easier_elemparts = elemparts

    return modules, graphs
