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

    # No matter the division is rounding up or down,
    # first (worldsize-1) workers always have per_worker_nv vertexes.
    per_worker_nv: int

    def get_start_end(self, rank=None):
        dist_env = get_runtime_dist_env()
        if rank == None:
            rank = dist_env.rank

        start = self.per_worker_nv * rank
        end = start + self.per_worker_nv
        if rank + 1 == dist_env.world_size:
            end = self.nv
        return start, end


@dataclass
class CoarseningLevel:
    """
    The outermost input graph is equal to a CoarseningLevel with all vertex
    weights being `1`.
    """
    dist_config: DistConfig

    # As we are merging vertexes and summing up their weights, this value
    # is simply the vertex number at the very begining and can be inherited
    # to all levels.
    total_vertex_weight: int

    # int(end-start,) weights for local vertexes
    vertex_weights: torch.Tensor

    rowptr: torch.Tensor
    colidx: torch.Tensor
    adjwgt: torch.Tensor

    cmap: torch.Tensor


def coarsen_level(
    prev_lv: CoarseningLevel,
    max_vertex_weight: int
) -> CoarseningLevel:
    """
    Do multi-level coarsening on all workers.
    On each level, it's run in a sequential manner from worker-0 to the last
    worker, with each worker coarsening its own part of adjacency matrix and
    broadcasting its matching result to others.

    Remarks:
    -   Given the nature of vertex degrees of the mesh, we don't sort vertexes
        by degrees first.
    -   As we aim to coarsen to certain level and then delegate
        deeper coarsening and partitioning to METIS, we only do
        heavy-edge matching (HEM) here, without 2-hop matching (i.e. match
        vertexes that both are neighbours of same vertex.)
    """
    dist_env = get_runtime_dist_env()

    rowptr = prev_lv.rowptr
    colidx = prev_lv.colidx
    rowwgt = prev_lv.rowptr
    adjwgt = prev_lv.adjwgt

    start, end = prev_lv.dist_config.get_start_end()
    assert rowptr.shape[0] -1 == end - start

    # max_vertex_weight = int(1.5 * total_vwgt // coarsen_to)
    
    # local vids to global vids
    matched = torch.full((end - start,), fill_value=-1, dtype=torch.int64)
    masked_colidx = colidx.clone()
    
    # old IDs to coarser IDs
    coarser_vid_map = torch.full((end - start,), fill_value=-1, dtype=torch.int64)

    cnv_allocated = 0

    for w in range(dist_env.world_size):
        if w == dist_env.rank:
            # int(?, 2): both values are global IDs, but the [:, 0] are held by this worker.
            matched_vid_pairs = torch.full((end - start, 2), fill_value=-1, dtype=torch.int64)

            # NOTE `matched` vector is updated within this C call
            n_new_matches = _C.locally_match_heavy_edge(
                start,
                end,
                matched,
                rowptr,
                masked_colidx,
                rowwgt,
                adjwgt,
                max_vertex_weight,
                matched_vid_pairs
            )
            matched_vid_pairs = matched_vid_pairs[:n_new_matches, :]

            assert torch.all(
                start < matched_vid_pairs[:, 1]
            ), "Never result in matching with vertexes on previous workers"

            ########
            # Broadcast the whole result, to mask out adj list for each worker.
            ########
            dist_env.broadcast_object_list(w, [matched_vid_pairs.shape[0]])
            w_matched_pairs = dist_env.broadcast(w, matched_vid_pairs)

            ########
            # Assign new vertex IDs for the coarser graph
            # and broadcast the ID assignments to where the vertexes are held
            ########
            unmatched_mask = matched == -1
            this_unmatched_n = int(unmatched_mask.count_nonzero())
            coarser_vid_map[unmatched_mask] = torch.arange(
                cnv_allocated,
                cnv_allocated + this_unmatched_n,
                dtype=torch.int64
            )

            # A matching will be stored twice, we only count once on the
            # less-than case (including two vertexes are both on this worker,
            # or the other is remote).
            this_assigned_to = matched_vid_pairs[:, 0]
            tell_cvids_to = matched_vid_pairs[:, 1]
            assert torch.all(
                this_assigned_to < tell_cvids_to
            ), \
                "Symmetric adjmat and adjw should always lead to" \
                "match with subsequent row"

            this_assigned_cvids = torch.arange(
                cnv_allocated,
                cnv_allocated + n_new_matches,
                dtype=torch.int64
            )
            coarser_vid_map[this_assigned_to - start] = this_assigned_cvids

            # Update coarser vid map for local part
            self_tell_mask = torch.logical_and(
                start <= tell_cvids_to,
                tell_cvids_to < end
            )
            coarser_vid_map[
                tell_cvids_to[self_tell_mask] - start
            ] = this_assigned_cvids[self_tell_mask]

            [
                w_last_cnv_allocated, w_assigned_n, w_unmatched_n 
            ] = dist_env.broadcast_object_list(w, [
                cnv_allocated, n_new_matches, this_unmatched_n
            ])

            w_tell_cvids_to = dist_env.broadcast(w, tell_cvids_to)

        else:
            # TODO all previous workers are involved in this broadcast, but
            [npairs] = dist_env.broadcast_object_list(w)
            w_matched_pairs = dist_env.broadcast(w, shape=(npairs, 2), dtype=torch.int64)

            # Till it reaches the point when this worker runs HEM matching,
            # it keeps overwriting these values the worker received.
            [ 
                w_last_cnv_allocated, w_assigned_n, w_unmatched_n
            ] = dist_env.broadcast_object_list(w)
            w_tell_cvids_to = dist_env.broadcast(w, shape=(w_assigned_n,), dtype=torch.int64)
        # end if w == rank

        cnv_allocated = w_last_cnv_allocated + w_assigned_n + w_unmatched_n

        if dist_env.rank > w:
            # Subsequent workers update their matched records
            to_this_mask = torch.logical_and(
                start <= w_matched_pairs[:, 1],
                w_matched_pairs[:, 1] < end
            )
            to_this_pairs = w_matched_pairs[to_this_mask]
            matched[to_this_pairs[:, 1] - start] = to_this_pairs[:, 0]

            # ... and mask out their adjmat cells
            masked_colidx[torch.isin(masked_colidx, w_matched_pairs)] = -1

            # Update coarser vid map for matching resulted by prev workers.
            w_tell_mask = torch.logical_and(
                start <= w_tell_cvids_to,
                w_tell_cvids_to < end
            )
            coarser_vid_map[
                w_tell_cvids_to[w_tell_mask] - start
            ] = torch.arange(
                w_last_cnv_allocated, w_last_cnv_allocated + w_assigned_n
            )[w_tell_cvids_to]
        # end if rank > w
    # end for w in range(world_size)
    matched = None
    masked_colidx = None
    cnv_allocated = None

    cnv: int = \
        w_last_cnv_allocated + w_assigned_n + w_unmatched_n  # type: ignore
    
    return merge_vertexes(prev_lv, cnv, coarser_vid_map)


def merge_vertexes(prev_lv: CoarseningLevel, cnv: int, cmap: torch.Tensor) -> CoarseningLevel:
    """
    Collectively merge vertexes into coarser vertexes, summing up their weights
    and unifying edges.

    Args:
    - cnv: the number of vertexes in the coarser graph, coarser vertexes
        are evenly distributed.
    - cmap: size of (end-start,) for old graph, mapping local vertex ID to
        new ID (in `range(cnv)`) in the coarser graph.
        To each new ID there are at most 2 old vertexes mapped.
    """
    dist_env = get_runtime_dist_env()

    c_per_worker_n = cnv // dist_env.world_size
    c_dist_config = DistConfig(cnv, c_per_worker_n)
    c_start, c_end = c_dist_config.get_start_end()

    ########
    # All2All to collect old vwgts for coarser vertexes
    ########
    rowptr = prev_lv.rowptr
    colidx = prev_lv.colidx
    rowwgt = prev_lv.rowptr
    adjwgt = prev_lv.adjwgt
    start, end = prev_lv.dist_config.get_start_end()

    cvids_unmerged = cmap[torch.arange(start, end)]
    cvids_unmerged_to_others = []
    vwgts_unmerged_to_others = []
    for w in range(dist_env.world_size):
        c_start_w, c_end_w = c_dist_config.get_start_end(w)
        mask_to_w = torch.logical_and(c_start_w <= cvids_unmerged, cvids_unmerged < c_end_w)

        cvids_unmerged_w = cvids_unmerged[mask_to_w]
        vwgts_unmerged_w = prev_lv.vertex_weights[mask_to_w]
        cvids_unmerged_to_others.append(cvids_unmerged_w)
        vwgts_unmerged_to_others.append(vwgts_unmerged_w)

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
    colidx_cmapped = torch.full_like(colidx, fill_value=-1)
    for w in range(dist_env.world_size):
        start_w, end_w = prev_lv.dist_config.get_start_end(w)
        if w == dist_env.rank:
            cmap_w = dist_env.broadcast(w, cmap)
        else:
            cmap_w = dist_env.broadcast(
                w, shape=(end_w - start_w,), dtype=torch.int64
            )

        # NOTE do not recursive manipulate adjmat, avoiding mixing vids
        # of old graph and coarser graph.
        mask_mappable_w = torch.logical_and(start_w <= colidx, colidx < end_w)
        assert torch.all(colidx_cmapped[mask_mappable_w] == -1, "no overlap")
        c_adj_unmerged = cmap_w[colidx[mask_mappable_w]]
        colidx_cmapped[mask_mappable_w] = c_adj_unmerged

    assert torch.all(colidx_cmapped[mask_mappable_w] != -1, "all mapped")

    # reuse cvids_unmerged = cmap[arange(start,end)]
    
        






def part_kway(
    rowptr, colidx, adjwgt
):
    pass