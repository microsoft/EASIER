# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union, cast
from dataclasses import dataclass
import torch.utils
from typing_extensions import Literal, OrderedDict, TypeAlias
import functools
import more_itertools
import time

import torch
from torch.fx.graph import Graph
from torch.fx.node import Node

import numpy as np
import scipy.sparse
from torch.nn.modules import Module

import easier.core.module as esr
from easier.core.runtime.dist_env import get_cpu_dist_env
from easier.core.utils import EasierJitException, logger
import easier.cpp_extension as _C



@dataclass
class DistConfig:
    nv: int
    local_nvs: List[int]

    @staticmethod
    def create_default(nv: int):
        # TODO use an incremental lengths sequence like [N, N+B, N+2B, ...]
        # since subsequent workers do less remote matching.
        dist_env = get_cpu_dist_env()
        per_worker_n, residue = divmod(nv, dist_env.world_size)
        local_nvs = [per_worker_n] * dist_env.world_size
        local_nvs[-1] += residue
        return DistConfig(nv, local_nvs)

    def get_start_end(self, rank=None):
        dist_env = get_cpu_dist_env()
        if rank == None:
            rank = dist_env.rank

        start = sum(self.local_nvs[:rank])
        end = start + self.local_nvs[rank]
        return start, end


@dataclass
class CoarseningLevel:
    """
    The outermost input graph is equal to a CoarseningLevel with all vertex
    weights being `1`.
    """
    # Coarser graph dist config of this level
    dist_config: DistConfig

    # As we are merging vertexes and summing up their weights, this value
    # is simply the vertex number at the very begining and can be inherited
    # to all levels.
    # total_vertex_weight: int

    # int(end-start,) weights for local vertexes of this level
    vertex_weights: torch.Tensor

    # CSR data for local adjmat of this level
    rowptr: torch.Tensor
    colidx: torch.Tensor
    adjwgt: torch.Tensor



def _assert_local_map_no_overlap(cvids, new_range):
    assert torch.all(cvids[new_range] == -1)

def assign_cvids_unmatched(
    matched: torch.Tensor, cvids: torch.Tensor, cnv_allocated: int
) -> int:
    # Vertexes of too big weights are skipped or not matched with
    unmatched_mask = matched == -1
    this_unmatched_n = int(unmatched_mask.count_nonzero())
    _assert_local_map_no_overlap(cvids, unmatched_mask)
    cvids[unmatched_mask] = torch.arange(
        cnv_allocated,
        cnv_allocated + this_unmatched_n,
        dtype=torch.int64
    )

    return this_unmatched_n

def assign_cvids_colocated(
    start: int, end: int,
    matched: torch.Tensor, cvids: torch.Tensor, cnv_allocated: int
) -> int:
    """
    "Colocated pairs" are match pairs whose vertexes are both on
    this worker, their cvids[x] values are the same.

    The assignment is decided on the preceding vertex,
    i.e. the matching invoker.

    Such vertexes will have no remoting matching, so here we are
    the first time processing and assigning coarser IDs for them.
    """
    colocated_mask = torch.logical_and(
        torch.arange(start, end) < matched,  # be matching invokers
        # not `arange() <=` because of no self-edge
        matched < end  # be in colocated pairs
    )
    colocated_from = torch.arange(start, end)[colocated_mask]
    colocated_to = matched[colocated_mask]
    this_colocated_n = colocated_from.shape[0]
    colocated_cvids = torch.arange(
        cnv_allocated,
        cnv_allocated + this_colocated_n,
        dtype=torch.int64
    )
    _assert_local_map_no_overlap(cvids, colocated_from - start)
    cvids[colocated_from - start] = colocated_cvids
    _assert_local_map_no_overlap(cvids, colocated_to - start)
    cvids[colocated_to - start] = colocated_cvids

    return this_colocated_n

def align_coarser_vids(
    remote_start: int, remote_end: int,
    remote_cvids: torch.Tensor,
    matched: torch.Tensor, cvids: torch.Tensor
):
    """
    For all remote vertexes (in subnsequent workers) to which
    the local vertexes are matched, gather the assigned new coarser IDs of
    the remote vertexes.
    And if a local vertex is matched to such a remote vertex, copy its coarser
    ID, so that both vertexes will be identified to be merged later.
    """
    remote_matched_mask = torch.logical_and(
        remote_start <= matched,
        matched < remote_end
    )
    _assert_local_map_no_overlap(cvids, remote_matched_mask)
    cvids.masked_scatter_(
        remote_matched_mask,
        remote_cvids[matched[remote_matched_mask] - remote_start]
    )


class CoarseningRowDataExchanger:
    # "Unmerged" means, given cvids a many-1 mapping, several old vertexes
    # are mapped to the same coarser vertex, we are yet to merge
    # old vertexes' weights and adj lists.
    def __init__(self, c_dist_config: DistConfig, cvids: torch.Tensor) -> None:
        dist_env = get_cpu_dist_env()
        c_start, c_end = c_dist_config.get_start_end()

        rows_to_other_masks = []
        cvids_unmerged_to_others = []
        for w in range(dist_env.world_size):
            c_start_w, c_end_w = c_dist_config.get_start_end(w)
            row_to_w_mask = torch.logical_and(
                c_start_w <= cvids, cvids < c_end_w
            )

            cvids_unmerged_to_w = cvids[row_to_w_mask]
            cvids_unmerged_to_others.append(cvids_unmerged_to_w)

            rows_to_other_masks.append(row_to_w_mask)

        self.rows_to_other_masks: List[torch.Tensor] = rows_to_other_masks

        # all in range [c_start, c_end), unordered and contains duplicates.
        self.cvids_unmerged_on_this = torch.concat(
            dist_env.all_to_all(cvids_unmerged_to_others)
        )

        self.c_dist_config = c_dist_config
        self.c_start = c_start
        self.local_cnv = c_end - c_start

    def exchange(self, row_data: torch.Tensor):
        dist_env = get_cpu_dist_env()
        row_data_to_others = []
        for w in range(dist_env.world_size):
            row_to_w_mask = self.rows_to_other_masks[w]
            row_data_to_w = row_data[row_to_w_mask]
            row_data_to_others.append(row_data_to_w)

        row_data_on_this = dist_env.all_to_all(row_data_to_others)
        return torch.concat(row_data_on_this)



def exchange_and_merge_vertex_weights(
    row_xchg: CoarseningRowDataExchanger,
    vertex_weights: torch.Tensor,
):
    """
    Use LOCAL cvids to determine destination and send vertex weights there.
    """
    cvwgts = torch.zeros((row_xchg.local_cnv,), dtype=torch.int64)

    vwgts_unmerged = row_xchg.exchange(vertex_weights)
    cvwgts.scatter_add_(
        dim=0,
        index=row_xchg.cvids_unmerged_on_this - row_xchg.c_start,
        src=vwgts_unmerged
    )
    assert torch.all(cvwgts > 0)
    return cvwgts

def exchange_cadj_adjw(
    row_xchg: CoarseningRowDataExchanger,
    rowptr: torch.Tensor,
    cadj: torch.Tensor,
    adjwgt: torch.Tensor
):
    """
    Use LOCAL cvids to determine destination and send data there.
    """
    assert cadj.shape[0] == adjwgt.shape[0]

    rowptr_begins = rowptr[:-1]
    rowptr_ends = rowptr[1:]
    row_sizes = rowptr_ends - rowptr_begins

    unmerged_row_sizes = row_xchg.exchange(row_sizes)

    # Because CSR does not have equal row substructure, we cannot directly
    # call `exchange`
    dist_env = get_cpu_dist_env()
    cadj_unmerged_to_others = []
    adjwgt_unmerged_to_others = []
    for w in range(dist_env.world_size):
        row_mask = row_xchg.rows_to_other_masks[w]
        col_mask = get_csr_mask_by_rows(rowptr, row_mask, cadj.shape[0])

        cadj_unmerged_to_w = cadj[col_mask]
        adjwgt_unmergedto_w = adjwgt[col_mask]

        cadj_unmerged_to_others.append(cadj_unmerged_to_w)
        adjwgt_unmerged_to_others.append(adjwgt_unmergedto_w)

    cadj_unmerged = torch.concat(
        dist_env.all_to_all(cadj_unmerged_to_others)
    )
    adjwgt_unmerged = torch.concat(
        dist_env.all_to_all(adjwgt_unmerged_to_others)
    )

    return unmerged_row_sizes, cadj_unmerged, adjwgt_unmerged


def merge_cadj_adjw(
    row_xchg: CoarseningRowDataExchanger,
    unmerged_row_sizes: torch.Tensor,
    cadj_unmerged: torch.Tensor,
    adjwgt_unmerged: torch.Tensor, 
):
    """
    Both are still CSR-formatted, the rowptr should be calculated from
    `unmerged_row_sizes`. E.g. for gathered CSR parts

    ```
    # of cvid 3, we get length-5 cadj list and adjw:
        [1 2 3 4 5] [1 1 1 1 1]
    # of cvid 7, we get length-4 cadj list and adjw:
        [3 5 8 9] [2 2 2 2]
    # ...
    ```

    we have:

    ```
    xchg.cvids_unmerged = [3 7]
    unmerged_row_sizes =  [5 4]
    unmerged_rows = [3 7].repeat_interleave([5 4])
                  = [3 3 3 3 3     7 7 7 7]  # to be x coords
    cadj_unmerged = [1 2 3 4 5] + [3 5 8 9]  # to be y coords
    adjw_unmerged = [1 1 1 1 1] + [2 2 2 2]  # to be graph weights
    ```
    """
    # NOTE either in a single `cvids_unmerged_from_w` or among all
    # `cvids_unmerged_on_this` these are duplicated coarser vids!
    # CANNOT be used directly as index or guide concating rowptr/colidx
    # e.g. given two mathced vertexes are mapped to the same coarser vertex,
    # these two vertexes may be on same workers,
    # we need to concat their adj lists, unique, and accumulate
    # edge weights accordingly.
    unmerged_rows = (
        row_xchg.cvids_unmerged_on_this - row_xchg.c_start
    ).repeat_interleave(
        unmerged_row_sizes
    )

    coarser_graph = scipy.sparse.csr_matrix(
        (adjwgt_unmerged, (unmerged_rows, cadj_unmerged)),  # sum up dups
        shape=(row_xchg.local_cnv, row_xchg.c_dist_config.nv)
    )

    lil = coarser_graph.tolil()
    lil.setdiag(0, row_xchg.c_start)  # remove self edge and weight
    coarser_graph = lil.tocsr()

    return torch.from_numpy(coarser_graph.indptr).to(torch.int64), \
        torch.from_numpy(coarser_graph.indices).to(torch.int64), \
        torch.from_numpy(coarser_graph.data).to(torch.int64)

def map_adj_by_cvids(
    dist_config: DistConfig, colidx: torch.Tensor, cvids: torch.Tensor
):
    """
    Use OTHERS' cvids mappings to map/transform LOCAL adj list.
    """
    dist_env = get_cpu_dist_env()

    cadj_unmerged = torch.full_like(colidx, fill_value=-1, dtype=torch.int64)
    for w in range(dist_env.world_size):
        start_w, end_w = dist_config.get_start_end(w)
        if w == dist_env.rank:
            cvids_w = dist_env.broadcast(w, cvids)
        else:
            cvids_w = dist_env.broadcast(
                w, shape=(end_w - start_w,), dtype=torch.int64
            )

        w_mappable = torch.logical_and(
            start_w <= colidx, colidx < end_w
        )
        _assert_local_map_no_overlap(cadj_unmerged, w_mappable)
        
        # "by_w" means its mappable part is mapped by cvids held by w, i.e.
        # whose domain is [start_w, end_w), but the codomain crosses workers.
        cadj_by_w = cvids_w[colidx[w_mappable] - start_w]

        cadj_unmerged[w_mappable] = cadj_by_w
    assert torch.all(cadj_unmerged[w_mappable] != -1), "all mapped"

    return cadj_unmerged


def get_csr_mask_by_rows(
    rowptr: torch.Tensor, row_mask: torch.Tensor, nnz: int
):
    """
    Args:
    - row_mask: 1 for rows to pick, 0 for rows to filter out.
    """
    rowptr_begins = rowptr[:-1]
    rowptr_ends = rowptr[1:]

    res_begins = rowptr_begins[row_mask]    
    res_ends = rowptr_ends[row_mask]

    # We'll skip the nnz-th item, as there is no more elements for the
    # rise signal to affect.
    col_rises = torch.zeros((nnz + 1,), dtype=torch.int8)
    col_rises[res_begins] = 1
    col_rises.scatter_add_(
        dim=0,
        index=res_ends,
        src=torch.tensor([-1], dtype=torch.int8).expand_as(res_ends)
    )
    col_levels = torch.cumsum(col_rises[:-1], dim=0, dtype=torch.int8)
    col_mask = col_levels == 1

    return col_mask


def merge_vertexes(
    prev_lv: CoarseningLevel, cnv: int, cvids: torch.Tensor
) -> CoarseningLevel:
    """
    Collectively merge vertexes into coarser vertexes, sum up their weights,
    merge their adj lists.

    E.g. to merge adj lists:
    Given cvids mapping { 2=>A, 5=>B, 6=>A } on this worker and mappings
    from other workers,
    we can map the adj mat:

    | 2 |  5  6  8  9
    | 5 |  9 11 15
    | 6 |  9 15

    to (cX are for other coarser vertexes)

    #cvids_unmerged
         #cadj_unmerged
    #~~~ #~~~~~~~~~~~    
    | A |  B  A c8 c9           # row_size = 4
    | B | c9 c1 c5
    | A | c9 c5                 # row_size = 2

    Then on the worker that coarser vertex A is located:

         # concat(cadj_on_this)
         #~~~~~~~~~~~~~~~~~
    | A |  B  A c8 c9 c9 c5     # row_size = 4+2

    By repeating coarser ID for row, A, by row_size times, and pair with
    coarser adjacent IDs, we get the COO data:

    [
        (A, B),
        (A, A),
        (A, c8),
        (A, c9),
        (A, c9),
        (A, c5),
    ]

    Then construct csr_matrix with the COO data, and remove self-edges:

    | A |  B c8 c9 c5

    Args:
    - cnv: the number of vertexes in the coarser graph, coarser vertexes
        are evenly distributed.
    - cvids: size of (end-start,) for old graph, mapping local vertex ID to
        new ID (in `range(cnv)`) in the coarser graph.
        To each new ID there may be many (even more than world_size)
        old vertexes mapped.
    """
    c_dist_config = DistConfig.create_default(cnv)
    xchg = CoarseningRowDataExchanger(c_dist_config, cvids)

    c_vwgts = exchange_and_merge_vertex_weights(xchg, prev_lv.vertex_weights)

    prev_lv_cadj = map_adj_by_cvids(
        prev_lv.dist_config, prev_lv.colidx, cvids
    )
    c_rows, c_cols, c_wgts = exchange_cadj_adjw(
        xchg, prev_lv.rowptr, prev_lv_cadj, prev_lv.adjwgt
    )
    c_rowptr, c_colidx, c_adjw = merge_cadj_adjw(xchg, c_rows, c_cols, c_wgts)

    c_lv = CoarseningLevel(
        c_dist_config,
        c_vwgts,
        c_rowptr,
        c_colidx,
        c_adjw,
    )
    return c_lv


def gather_csr_graph(dst_rank: int, clv: CoarseningLevel
) -> Optional[Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]]:
    """
    Gather all pieces from all workers, reconstruct them into a valid
    individual CSR data.
    """
    dist_env = get_cpu_dist_env()
    vwgts = dist_env.gather(dst_rank, clv.vertex_weights)
    rowptrs = dist_env.gather(dst_rank, clv.rowptr)
    colidxs = dist_env.gather(dst_rank, clv.colidx)
    adjws = dist_env.gather(dst_rank, clv.adjwgt)
    if dist_env.rank == dst_rank:
        assert vwgts is not None
        assert rowptrs is not None
        assert colidxs is not None
        assert vwgts is not None
        assert adjws is not None

        colidx_sizes = [int(rp[-1]) for rp in rowptrs]
        bases = torch.tensor([0] + colidx_sizes[:-1], dtype=torch.int64)
        bases = torch.cumsum(bases, dim=0)

        for i in range(dist_env.world_size):
            rowptr_i = rowptrs[i] + bases[i]
            if i + 1 < dist_env.world_size:
                rowptr_i = rowptr_i[:-1]
            rowptrs[i] = rowptr_i

        res_vwgt = torch.concat(vwgts)
        res_rowptr = torch.concat(rowptrs)
        res_colidx = torch.concat(colidxs)
        res_adjw = torch.concat(adjws)

        return res_vwgt, res_rowptr, res_colidx, res_adjw
    else:
        return None


def coarsen_level(
    prev_lv: CoarseningLevel
) -> Tuple[CoarseningLevel, torch.Tensor]:

    dist_env = get_cpu_dist_env()

    # TODO make later workers have more rows to process
    start, end = prev_lv.dist_config.get_start_end()
    assert prev_lv.rowptr.shape[0] -1 == end - start

    ########
    # Each worker independently calculates heavy-edge matching
    ########
    # local vids to global vids
    matched = torch.full((end - start,), fill_value=-1, dtype=torch.int64)
    # NOTE `matched` vector is updated within this C call.
    _C.locally_match_heavy_edge(
        start,
        end,
        matched,
        prev_lv.rowptr,
        prev_lv.colidx,
        prev_lv.adjwgt,
    )
    # Possible value of matched[x]:
    # -1 
    #   unmatched
    # end <= matched[x]
    #   matched with remote vertexes
    # x + start < matched[x] < end
    #   matching invoker, matched with local vertexes (colocated)
    # start <= matched[x] < start + x
    #   matched-with vertexes, colocated
    
    cnv_allocated = 0  # replicated

    # Old local IDs of owned vertexes to coarser IDs
    cvids = torch.full((end - start,), fill_value=-1, dtype=torch.int64)

    for w in range(dist_env.world_size - 1, -1, -1):
        w_start, w_end = prev_lv.dist_config.get_start_end(w)

        if w == dist_env.rank:
            this_unmatched_n = assign_cvids_unmatched(
                matched, cvids, cnv_allocated
            )
            cnv_allocated += this_unmatched_n

            this_colocated_n = assign_cvids_colocated(
                start, end, matched, cvids, cnv_allocated
            )
            cnv_allocated += this_colocated_n

            # NOTE Remaining `matched` elements are local vertexes
            # that are matching with remote vertexes (on subsequent workers),
            # they are processed in the `if rank < w:` part beblow in previous
            # iterations of those subsequent workers.
            assert torch.all(cvids != -1), \
                "All local vertexes should be assigned with coarser IDs"

            # TODO make a masked broadcast for only (rank < w) workers.
            w_cvids = dist_env.broadcast(w, cvids)

            [cnv_allocated] = dist_env.broadcast_object_list(
                w, [cnv_allocated]
            )

        else:
            w_cvids = dist_env.broadcast(
                w, shape=(w_end - w_start,), dtype=torch.int64
            )
            [cnv_allocated] = dist_env.broadcast_object_list(w)
        # end if rank == w

        if dist_env.rank < w:
            align_coarser_vids(w_start, w_end, w_cvids, matched, cvids)

    # end for w in range(world_size)

    new_lv = merge_vertexes(
        prev_lv, cnv_allocated, cvids
    )
    return new_lv, cvids




def distpart_kway(
    dist_config: DistConfig,
    rowptr: torch.Tensor,
    colidx: torch.Tensor,
    adjwgt: torch.Tensor,
):
    dist_env = get_cpu_dist_env()
    local_nv = dist_config.local_nvs[dist_env.rank]

    cur_lv = CoarseningLevel(
        dist_config=dist_config,
        rowptr=rowptr.to(torch.int64),
        colidx=colidx.to(torch.int64),
        # At the beginning, all vertex weights are 1
        vertex_weights=torch.ones((local_nv,), dtype=torch.int64),
        adjwgt=adjwgt.to(torch.int64)
    )

    # TODO because we use simple, directedly mapped uncoarsening, without
    # refinement, we don't need store the CoarseningLevel data.
    # levels: List[CoarseningLevel] = []
    c_dist_configs: List[DistConfig] = []

    # For CoarseningLevel-i to CoarsenLevel-(i+1), the cvids is stored in
    # level (i+1).
    # The length of cvids is the local vertex number for previous level,
    # the value of cvids is the global ID of coarser vertex in this level. 
    cvids_levels: List[torch.Tensor] = []

    coarsening_start = time.time()
    logger.debug(f"EASIER coarsening started. nv={dist_config.nv}")
    
    while True:
        new_lv, cvids = coarsen_level(cur_lv)
        # TODO levels.append(new_lv)
        c_dist_configs.append(new_lv.dist_config)
        cvids_levels.append(cvids)

        logger.debug(f"New level coarsened. nv={new_lv.dist_config.nv}")

        cur_nv = cur_lv.dist_config.nv
        new_nv = new_lv.dist_config.nv
        if (cur_nv - new_nv) < int(cur_nv * 0.2):
            break
        if new_nv < 10 ** 5:
            break

        cur_lv = new_lv

    coarsening_latency = time.time() - coarsening_start
    logger.debug(
        f"EASIER coarsening finished. Total time: {coarsening_latency}sec"
    )

    # Gather to worker-0 and call METIS
    cgraph = gather_csr_graph(0, new_lv)
    if dist_env.rank == 0:
        assert cgraph is not None
        vwgt0, rowptr0, colidx0, adjw0 = cgraph

        from mgmetis import metis
        ncuts, membership = metis.part_graph_kway(
            nparts=dist_env.world_size,
            xadj=rowptr0,
            adjncy=colidx0,
            vwgt=vwgt0,
            adjwgt=adjw0
        )
        logger.debug(
            f"METIS result on EASIER-coarsened graph: ncuts={ncuts}"
        )

        # TODO scatter tensor list API
        c_local_membership = dist_env.scatter_object(
            0,
            torch.from_numpy(membership).to(torch.int64).split(
                new_lv.dist_config.local_nvs
            )
        )
    
    else:
        c_local_membership = dist_env.scatter_object(0)

    # Uncoarsening
    # TODO now without refinment (i.e. move vertexes around partitions after
    # uncoarsening one level, try achieving better global partition quality)
    for i in range(len(cvids_levels) - 1, -1, -1):
        cvids = cvids_levels[i]
        c_dist_config = c_dist_configs[i]

        local_membership = uncoarsen_level(c_dist_config, c_local_membership, cvids)

        c_local_membership = local_membership

    assert local_membership.shape[0] == local_nv

    distpart_latency = time.time() - coarsening_start
    logger.debug(
        f"EASIER partition finished. Total time: {distpart_latency}sec"
    )

    # Returns local_membership for vertexes of the original input graph
    return local_membership

def uncoarsen_level(c_dist_config: DistConfig, c_local_membership: torch.Tensor, cvids: torch.Tensor):
    """
    To uncoarsen level-(i+1) back to level-i, the cvids length is about
    local vertexes in level-i, its values are level-(i+1) vertex IDs.
    """
    dist_env = get_cpu_dist_env()
    local_membership = torch.full((cvids.shape[0],), fill_value=-1, dtype=torch.int64)

    for w in range(dist_env.world_size):
        c_start_w, c_end_w = c_dist_config.get_start_end(w)
        if w == dist_env.rank:
            c_local_membership_w = dist_env.broadcast(
                w, c_local_membership
            )
        else:
            c_local_membership_w = dist_env.broadcast(
                w, shape=(c_end_w - c_start_w,), dtype=torch.int64
            )
        
        inv_mask = torch.logical_and(c_start_w <= cvids, cvids < c_end_w)
        _assert_local_map_no_overlap(local_membership, inv_mask)
        local_membership[inv_mask] = c_local_membership_w[cvids[inv_mask] - c_start_w]
        
    assert torch.all(local_membership != -1)
    return local_membership
