# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union, cast
from dataclasses import dataclass
import torch.utils
from typing_extensions import Literal, OrderedDict, TypeAlias
import functools
import more_itertools

import torch
from torch.fx.graph import Graph
from torch.fx.node import Node

import numpy as np
import scipy.sparse
from torch.nn.modules import Module

import easier.core.module as esr
from easier.core.passes.utils import \
    EasierInterpreter, OrderedSet, \
    get_selector_reducer_idx_partition_pair, \
    normalize_reducer_call_into_args, normalize_selector_call_into_args, \
    get_easier_tensors
from easier.core.runtime.dist_env import get_runtime_dist_env
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
        dist_env = get_runtime_dist_env()
        per_worker_n, residue = divmod(nv, dist_env.world_size)
        local_nvs = [per_worker_n] * dist_env.world_size
        local_nvs[-1] += residue
        return DistConfig(nv, local_nvs)

    def get_start_end(self, rank=None):
        dist_env = get_runtime_dist_env()
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

def get_adj_vwgt():
    """
    Given the max_vertex_weight constraint, if the sum of two 
    """
    pass

def gather_csr_graph(dst_rank: int, clv: CoarseningLevel
) -> Optional[Tuple[
    List[int],
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]]:
    """
    Gather all pieces from all workers, reconstruct them into a valid
    individual CSR data.
    """
    dist_env = get_runtime_dist_env()
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

        sizes = [int(rp[-1]) for rp in rowptrs]
        bases = torch.tensor([0] + sizes[:-1], dtype=torch.int64)
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

        return sizes, res_vwgt, res_rowptr, res_colidx, res_adjw
    else:
        return None



def coarsen_level(
    prev_lv: CoarseningLevel, max_vertex_weight: int
) -> Tuple[CoarseningLevel, torch.Tensor]:

    dist_env = get_runtime_dist_env()

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
        prev_lv.vertex_weights,
        prev_lv.adjwgt,
        max_vertex_weight
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
    
    ########
    # Assign new vertex IDs for the coarser graph
    # xxxx and broadcast the ID assignments to where the vertexes are held
    ########
    cnv_allocated = 0  # replicated

    # Old local IDs of owned vertexes to coarser IDs
    coarser_vid_map = torch.full((end - start,), fill_value=-1, dtype=torch.int64)
    def _assert_cmap_no_overlap(new_range):
        assert torch.all(coarser_vid_map[new_range] == -1)

    for w in range(dist_env.world_size - 1, -1, -1):
        w_start, w_end = prev_lv.dist_config.get_start_end(w)

        if w == dist_env.rank:
            # Vertexes of too big weights are skipped or not matched with
            unmatched_mask = matched == -1
            this_unmatched_n = int(unmatched_mask.count_nonzero())
            _assert_cmap_no_overlap(unmatched_mask)
            coarser_vid_map[unmatched_mask] = torch.arange(
                cnv_allocated,
                cnv_allocated + this_unmatched_n,
                dtype=torch.int64
            )

            # "Colocated pairs" are match pairs whose vertexes are both on
            # this worker, their cmap[x] values are the same.
            #
            # Such vertexes will have no remoting matching, so here we are
            # the first time processing and assigning coarser IDs for them.
            colocated_mask = torch.logical_and(
                matched < end,  # be in colocated pairs
                torch.arange(start, end) < matched  # be matching invokers
                # not `arange() <=` because of no self-edge
            )
            colocated_from_lvid = torch.arange(start, end)[colocated_mask]
            colocated_to_gvid = matched[colocated_mask]
            this_colocated_n = colocated_from_lvid.shape[0]
            colocated_cvids = torch.arange(
                cnv_allocated + this_unmatched_n,
                cnv_allocated + this_unmatched_n + this_colocated_n,
                dtype=torch.int64
            )
            _assert_cmap_no_overlap(colocated_from_lvid)
            coarser_vid_map[colocated_from_lvid] = colocated_cvids
            _assert_cmap_no_overlap(colocated_to_gvid - start)
            coarser_vid_map[colocated_to_gvid - start] = colocated_cvids

            # NOTE Remaining `matched` elements are local vertexes
            # that are matching with remote vertexes (on subsequent workers),
            # they are processed in the `if rank < w:` part beblow in previous
            # iterations of those subsequent workers.
            assert torch.all(coarser_vid_map != -1), \
                "All local vertexes should be assigned with coarser IDs"

            # TODO make a masked broadcast for only (rank < w) workers.
            w_cmap = dist_env.broadcast(w, coarser_vid_map)

            [cnv_allocated] = dist_env.broadcast_object_list(w, [
                cnv_allocated + this_unmatched_n + this_colocated_n
            ])

        else:
            w_cmap = dist_env.broadcast(w, shape=(w_end - w_start,), dtype=torch.int64)

            [cnv_allocated] = dist_env.broadcast_object_list(w)
        # end if rank == w

        if dist_env.rank < w:
            # Align with w's cvids
            remote_matched_on_w_mask = torch.logical_and(
                w_start <= matched,
                matched < w_end
            )
            _assert_cmap_no_overlap(remote_matched_on_w_mask)
            coarser_vid_map.masked_scatter_(
                remote_matched_on_w_mask,
                w_cmap[matched[remote_matched_on_w_mask]]
            )
        # end if rank < w
    # end for w in range(world_size)

    new_lv = merge_vertexes(
        prev_lv, cnv_allocated, coarser_vid_map
    )
    return new_lv, coarser_vid_map

def merge_vertexes(
    prev_lv: CoarseningLevel, cnv: int, cmap: torch.Tensor
) -> CoarseningLevel:
    """
    Collectively merge vertexes into coarser vertexes, sum up their weights,
    merge their adj lists.

    E.g.
    Given cmap { 2=>A, 5=>B, 6=>A } on this worker and many cmaps
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
    - cmap: size of (end-start,) for old graph, mapping local vertex ID to
        new ID (in `range(cnv)`) in the coarser graph.
        To each new ID there may be many (even more than world_size)
        old vertexes mapped.
    """
    dist_env = get_runtime_dist_env()

    c_per_worker_n = cnv // dist_env.world_size
    c_dist_config = DistConfig.create_default(cnv)
    c_start, c_end = c_dist_config.get_start_end()

    ########
    # All2All to collect old vwgts for coarser vertexes
    ########
    start, end = prev_lv.dist_config.get_start_end()

    # Collect mapped coarser vertexes to where they are located (in a sense of
    # coarser graph).
    #
    # "Unmerged" means, given cmap is a many-1 mapping, several old vertexes
    # are mapped to the same coarser vertex, we are yet to merge
    # old vertexes' weights and adj lists.
    cvids_unmerged_to_others = []
    vwgts_unmerged_to_others = []
    for w in range(dist_env.world_size):
        c_start_w, c_end_w = c_dist_config.get_start_end(w)
        row_coarsened_to_w_mask = torch.logical_and(
            c_start_w <= cmap, cmap < c_end_w
        )

        cvids_unmerged_to_w = cmap[row_coarsened_to_w_mask]
        vwgts_unmerged_to_w = prev_lv.vertex_weights[row_coarsened_to_w_mask]
        cvids_unmerged_to_others.append(cvids_unmerged_to_w)
        vwgts_unmerged_to_others.append(vwgts_unmerged_to_w)

    cvids_unmerged_on_this = dist_env.all_to_all(cvids_unmerged_to_others)
    vwgts_unmerged_on_this = dist_env.all_to_all(vwgts_unmerged_to_others)

    cvwgts = torch.zeros((c_end - c_start,), dtype=torch.int64)
    for w in range(dist_env.world_size):
        cvwgts.scatter_add_(
            dim=0,
            index=cvids_unmerged_on_this[w] - c_start,
            src=vwgts_unmerged_on_this[w]
        )
    assert torch.all(cvwgts > 0)

    ########
    # Broadcast all cmaps, each worker uses it to map its adjmat CSR,
    # then All2All to store new adjmat along with coarser vertexes.
    ########
    cadj_unmerged = torch.full_like(prev_lv.colidx, fill_value=-1, dtype=torch.int64)
    for w in range(dist_env.world_size):
        start_w, end_w = prev_lv.dist_config.get_start_end(w)
        if w == dist_env.rank:
            cmap_w = dist_env.broadcast(w, cmap)
        else:
            cmap_w = dist_env.broadcast(
                w, shape=(end_w - start_w,), dtype=torch.int64
            )

        cmap_w_mappable = torch.logical_and(
            start_w <= prev_lv.colidx, prev_lv.colidx < end_w
        )
        assert torch.all(cadj_unmerged[cmap_w_mappable] == -1), "no overlap"
        
        # "by_w" means its mappable part is mapped by cmap held by w, i.e.
        # whose domain is [start_w, end_w), but the codomain crosses workers.
        cadj_by_w = cmap_w[prev_lv.colidx[cmap_w_mappable]]

        cadj_unmerged[cmap_w_mappable] = cadj_by_w
    assert torch.all(cadj_unmerged[cmap_w_mappable] != -1), "all mapped"

    row_sizes_to_others = []
    cadj_unmerged_to_others = []
    adjw_unmerged_to_others = []


    for w in range(dist_env.world_size):
        c_start_w, c_end_w = c_dist_config.get_start_end(w)
        # TODO this mask is calculated again, maybe we can merge this loop with
        # the above loop, but balancing clarity.
        row_coarsened_to_w_mask = torch.logical_and(
            c_start_w <= cmap, cmap < c_end_w
        )
        to_w_rowptr_begins = prev_lv.rowptr[:-1][row_coarsened_to_w_mask]
        to_w_rowptr_ends = prev_lv.rowptr[1:][row_coarsened_to_w_mask]

        to_w_row_sizes = to_w_rowptr_ends - to_w_rowptr_begins

        col_mask = get_csr_mask_by_rows(
            to_w_rowptr_begins, to_w_rowptr_ends, cadj_unmerged.shape[0]
        )
        cadj_unmerged_to_w = cadj_unmerged[col_mask]
        adjw_unmergedto_w = prev_lv.adjwgt[col_mask]

        row_sizes_to_others.append(to_w_row_sizes)
        cadj_unmerged_to_others.append(cadj_unmerged_to_w)
        adjw_unmerged_to_others.append(adjw_unmergedto_w)

    # reuse cvids_unmerged_on_this, the All2All result
    cvids_unmerged_on_this: List[torch.Tensor]

    row_sizes_on_this = dist_env.all_to_all(row_sizes_to_others)
    cadj_unmerged_on_this = dist_env.all_to_all(cadj_unmerged_to_others)
    adjw_unmerged_on_this = dist_env.all_to_all(adjw_unmerged_to_others)

    # NOTE either in a single `cvids_unmerged_from_w` or among all
    # `cvids_unmerged_on_this` these are duplicated coarser vids!
    # CANNOT be used directly as index or guide concating rowptr/colidx
    # e.g. given two mathced vertexes are mapped to the same coarser vertex,
    # these two vertexes may be on same workers,
    # we need to concat their adj lists, unique, and accumulate
    # edge weights accordingly.
    xs_pieces = []
    for w in range(dist_env.world_size):
        cvids_unmerged_submat = cvids_unmerged_on_this[w] - c_start
        row_sizes_submat = row_sizes_on_this[w]
        xs_pieces.append(
            torch.repeat_interleave(cvids_unmerged_submat, row_sizes_submat)
        )
    xs = torch.concat(xs_pieces)
    ys = torch.concat(cadj_unmerged_on_this)
    ws = torch.concat(adjw_unmerged_on_this)
    height = c_end - c_start
    width = c_dist_config.nv

    coarser_graph = scipy.sparse.csr_matrix(
        (ws, (xs, ys)),  # sum up dups
        shape=(height, width)
    )

    g = coarser_graph.tolil()
    g.setdiag(0, c_start)  # remove self edge and weight
    coarser_graph = g.tocsr()

    c_rowptr = torch.from_numpy(coarser_graph.indptr)
    c_colidx = torch.from_numpy(coarser_graph.indices)
    c_adjw = torch.from_numpy(coarser_graph.data)

    c_lv = CoarseningLevel(
        c_dist_config,
        cvwgts,
        c_rowptr,
        c_colidx,
        c_adjw,
    )
    return c_lv


def get_csr_mask_by_rows(rowptr: torch.Tensor, row_mask: torch.Tensor, nnz: int):
    """
    Args:
    - row_mask: 1 for rows to pick, 0 for rows to filter out.
    """
    rowptr_begins = rowptr[:-1]
    rowptr_ends = rowptr[1:]

    res_begins = rowptr_begins[row_mask]    
    res_ends = rowptr_ends[row_mask]

    col_rises = torch.zeros((nnz,), dtype=torch.int8)
    col_rises[res_begins] = 1
    col_rises.scatter_add_(
        dim=0,
        index=res_ends,
        src=torch.tensor([-1], dtype=torch.int8).expand((nnz,))
    )
    col_levels = torch.cumsum(col_rises, dim=0, dtype=torch.int8)
    col_mask = col_levels == 1

    return col_mask


def distpart_kway(
    dist_config: DistConfig,
    rowptr: torch.Tensor,
    colidx: torch.Tensor,
    adjwgt: torch.Tensor,
):
    dist_env = get_runtime_dist_env()
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

    # For CoarseningLevel-i to CoarsenLevel-(i+1), the cmap is stored in
    # level (i+1).
    # The length of cmap is the local vertex number for previous level,
    # the value of cmap is the global ID of coarser vertex in this level. 
    cmaps: List[torch.Tensor] = []

    for i in range(5):  # TODO fake
        new_lv, cmap = coarsen_level(cur_lv, 10000)
        # TODO levels.append(new_lv)
        c_dist_configs.append(new_lv.dist_config)
        cmaps.append(cmap)
        cur_lv = new_lv

    # Gather to worker-0 and call METIS
    cgraph = gather_csr_graph(0, new_lv)
    if dist_env.rank == 0:
        assert cgraph is not None
        nvs, vwgt0, rowptr0, colidx0, adjw0 = cgraph
        nv0 = int(rowptr0[-1])

        from mgmetis import metis
        ncuts, membership = metis.part_graph_kway(
            1, rowptr0, colidx0,
            adjwgt=adjw0
        )

        # TODO scatter tensor list API
        c_local_membership = dist_env.scatter_object(
            0,
            torch.from_numpy(membership).split(nvs)
        )
    
    else:
        c_local_membership = dist_env.scatter_object(0)

    # Uncoarsening
    # TODO now without refinment (i.e. move vertexes around partitions after
    # uncoarsening one level, try achieving better global partition quality)
    for i in range(len(cmaps) - 1, -1, -1):
        cmap = cmaps[i]
        c_dist_config = c_dist_configs[i]

        # lv.cmap's domain is the set of finer vertexes on the same worker
        # (i.e. for the next uncoarsening level)
        #
        # Its codomain is the set of coarser vertexes across workers, which is
        # also the incoming local_membership is about,
        # as we are now reversing the coarsening map.

        # For next uncoarsening level
        local_membership = torch.full((cmap.shape[0],), fill_value=-1, dtype=torch.int64)

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
            
            inv_mask = torch.logical_and(c_start_w <= cmap, cmap < c_end_w)
            local_membership[inv_mask] = c_local_membership_w[cmap[inv_mask] - c_start_w]
            
        assert torch.all(local_membership != -1)
        c_local_membership = local_membership

    assert local_membership.shape[0] == local_nv

    # Returns local_membership for vertexes of the original input graph
    return local_membership


def part_kway(
    dist_config: DistConfig,
    rowptr: torch.Tensor,
    colidx: torch.Tensor,
    adjwgt: torch.Tensor,
):
    import os
    if os.environ.get('PARTITION_METHOD', 'DISTPART').upper() == 'DISTPART':
        return distpart_kway(dist_config, rowptr, colidx, adjwgt)
    
    else:
        from mpi4py import MPI
        from mgmetis import parmetis
        comm: MPI.Intracomm = MPI.COMM_WORLD
        ncuts, local_membership = parmetis.part_kway(
            comm.size,
            rowptr,
            colidx,
            vtxdist=torch.tensor(
                [0] + dist_config.local_nvs, dtype=torch.int64
            ).cumsum(dim=0),
            comm=comm,
            adjwgt=adjwgt
        )

        return local_membership