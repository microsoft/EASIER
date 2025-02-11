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



MASKVALUE = -1

@dataclass
class CoarseningLevel:
    orphans: torch.Tensor




class DistSHEMMatcher:
    """
    Match vertexes (strictly into 2-vertex pairs) in a distributed manner,
    using SHEM (sorted heavy-edge matching) algorithm.

    This requires all vertexes are globally sorted by their degrees.
    TODO later workers have longer adj list therefore more memory, we may adaptively
    adjust the size of each worker.
    """
    def __init__(self):
        # # The CSR data for the adjacency matrix of size (N/worldsize, N) for
        # # the partition on this worker.
        # self.rowptr: torch.Tensor
        # self.colidx: torch.Tensor
        # self.adj_weights: torch.Tensor

        # Padded adjacency lists (a list of adjacent vertex IDs) padded with -1
        # of size (N/worldsize, M) -- M is the global max degree
        self.adj_lists: torch.Tensor

        # Ints, of size (N/worldsize, M), padded with -1
        self.adj_weights: torch.Tensor

        # Ints of (N/worldsize,) whose values are the global IDs of the rows,
        # in case rows are shuffled/sorted or filtered.
        self.vertexes_ids: torch.Tensor
        self.vertex_weights: torch.Tensor


        # Ints of (N/worldsize,) whose values are the global IDs of
        # matched vertexes. The default value is -1.
        self.matched: torch.Tensor


        # adj lists that is getting masked out
        self.masked_adj_lists: torch.Tensor
        self.masked_adj_weights: torch.Tensor
        
    def core_hem(self, max_vertex_weight=1):
        """
        The core of the HEM process using vectorized ops. This process should
        be done on one worker at a time and each worker run once
        in a sequential manner.

        Preconditions:
        -   Several vertexes/rows of `adj_lists/adj_weights/vertexes_ids`
            stored on this worker are deactivated,
            because they have been matched in the previous turns;
        -   Several adjacent cells in `adj_lists/adj_weights` are masked out,
            including and more than the aforementioned vertexes/rows,
            because they have been matched in the previous turns;
        
        Algorithm:


        Remarks:
        -   As coarsening goes, vertex weights may be >=1 because of accumulation.
        -   It's possible that the adjacency info of even a single 2-vertex pair
            is not collected in one turn of matching. We'll match using what info
            we have and then broadcast the matching result.
        """

        # To avoid cyclic matching, when an adjacent vertex is also
        # a vertex (row) to process, we enforce that vertex (row) can
        # only match with such adjacent vertexes if the row's ID is less than
        # the ID of the adjacent vertex.
        # I.e. we mask out cells (adj_in_vids && adj <= vid) to -inf.
        cyclic_mask = torch.logical_and(
            torch.isin(self.masked_adj_lists, self.vertexes_ids),
            self.masked_adj_lists <= self.vertexes_ids[:, None]
        )
        weight_constraint_mask = (
            self.masked_adj_weights + self.vertex_weights[:, None]
        ) <= max_vertex_weight

        adj_mask = torch.logical_or(cyclic_mask, weight_constraint_mask)

        matched = self.matched.clone()
        adjmat = self.masked_adj_lists.clone()
        adjw = self.masked_adj_weights.clone()

        adjmat[adj_mask] = MASKVALUE
        adjw[adj_mask] = MASKVALUE

        for _ in range(self.vertexes_ids.shape[0]):
            # bool(V,)  s.t. nonzero() == P
            row_flags = torch.logical_and(matched == MASKVALUE, matched < max_vertex_weight)

            # int(P,)
            unmatched_vids = self.vertexes_ids[row_flags]

            if unmatched_vids.shape[0] == 0:
                # No more vertexes eligible for processing
                break

            # int(P, MaxDegree)
            unmatched_adjmat = adjmat[row_flags, :]
            unmatched_adjw = adjw[row_flags, :]

            # TODO
            # the worst case is that all (V-1) rows are depended, this loop
            # degrades to a totally O(V) complexity. We may shuffle and divide
            # all rows into two sets and loop twice to break dependency.
            # And depending on how large A is, shuffle and divide again. 

            # bool(P,)  s.t. nonzero() == A
            active_row_noref_flags = ~torch.isin(unmatched_vids, unmatched_adjmat)

            # int(A,)
            active_vids = unmatched_vids[active_row_noref_flags]

            # bool(A, MaxDegree)
            active_adjmat = unmatched_adjmat[active_row_noref_flags, :]            
            active_adjw = unmatched_adjw[active_row_noref_flags, :]

            heavy_adjw, adj_pos = torch.max(active_adjw, dim=1)
            adj_vids = active_adjmat[range(active_adjw.shape[0]), adj_pos]

            # TODO unique adj_vids by smaller rows
            torch.unique(adj_vids, returns_inverse=True)

            # heavy_adjw may be -1 (e.g. all matching are overweight),
            # we leave this vertex as unmatched, by temporarily setting the
            # matched ID to itself.
            matched_vids = torch.where(heavy_adjw == MASKVALUE, active_vids, adj_vids)
            matched[active_vids] = matched_vids

        else:
            assert False

        self.matched = matched


    def match_heavy_edge(self, rows: torch.Tensor):
        """
        The pipeline parallelism

        Args:
        -   rows: Row indexes, also the vertex IDs of this coarsening level. 
                Not ordered (because vertexes are partitioned, and are firstly sorted by degrees)
        """
        dist_env = get_runtime_dist_env()

        for w in range(dist_env.world_size):
            if w == dist_env.rank:
                self.core_hem()

            self.sync_matched(w)

        # Reset all temporary self-matching to be unmatched
        self_matching_flags = self.matched == self.vertexes_ids
        self.matched[self_matching_flags] = MASKVALUE

    
    def sync_matched(self, src_rank):
        """
        Update all workers:
        -   the `matched` array
        -   the `masked_adj_xxx` matrix
        """
        dist_env = get_runtime_dist_env()

        if src_rank == dist_env.rank:
            dist_env.broadcast_object_list(src_rank, [self.vertexes_ids.shape[0]])
            dist_env.broadcast(src_rank, self.vertexes_ids)
            dist_env.broadcast(src_rank, self.matched)
        else:
            [nv] = dist_env.broadcast_object_list(src_rank)
            w_vids = dist_env.broadcast(src_rank, shape=[nv], dtype=torch.int64)
            w_matched = dist_env.broadcast(src_rank, shape=[nv], dtype=torch.int64)


    def core_2hop_connected(self, row_flags):
        """
        No adjweight considered.
        Working on the new column-wise adjmat.
        """
        adjmat = self.masked_adj_lists

        hop_adjmat = adjmat[row_flags, :]
        n_active = hop_adjmat.shape[0]
        
        # on each active row, we try to match the fareast two adj
        # then we deduplicate among all independent matching results
        
        # argmax returns the first pos if there are multiple maximals
        lb_pos = torch.argmax(hop_adjmat != MASKVALUE, dim=1)
        ub_pos = -torch.argmax((hop_adjmat != MASKVALUE).fliplr(), dim=1) - 1

        unmatched_vids_lb = hop_adjmat[:, lb_pos]
        unmatched_vids_ub = hop_adjmat[:, ub_pos]
        # two vectors are zipped to form pairs, and all vids in those pairs
        # are required to appear at most once.

        flags = unmatched_vids_lb != unmatched_vids_ub
        unmatched_vids_lb = unmatched_vids_lb[flags]
        unmatched_vids_ub = unmatched_vids_ub[flags]

        # Because of vectorized operations, we don't require a strict
        # resolution order of row-by-row as the serial version of algo.
        def _unique_one_side(sort_side: torch.Tensor, follow_side: torch.Tensor):
            sorted, inv = sort_side.unique(sorted=True,  return_inverse=True)
            follow = torch.empty((sorted.shape[0],), dtype=torch.int64)

            # Undeterministically overwrite, we'll take whatever IDs win
            follow[inv] = follow_side
            return sorted, follow

        ub_vids, lb_vids = _unique_one_side(unmatched_vids_ub, unmatched_vids_lb)
        lb_vids, ub_vids = _unique_one_side(lb_vids, ub_vids)

        picked_lbs = []
        picked_ubs = []
        picked_vids = torch.empty((0,), dtype=torch.int64)

        while True:
            # like progressively pick non-source part of a directed graph.
            flags = ~torch.isin(lb_vids, ub_vids)
            picked_lb_vids = lb_vids[flags]
            picked_ub_vids = ub_vids[flags]

            picked_vids = torch.concat([picked_vids, picked_lb_vids, picked_ub_vids]).unique()

            picked_lbs.append(picked_lb_vids)
            picked_ubs.append(picked_ub_vids)

            lb_vids = lb_vids[~flags]
            ub_vids = ub_vids[~flags]

            mask = torch.logical_or(torch.isin(lb_vids, picked_vids), torch.isin(ub_vids, picked_vids))
            lb_vids = lb_vids[~mask]
            ub_vids = ub_vids[~mask]

            if lb_vids.shape[0] == 0:
                break

        torch.concat(picked_lbs)
        torch.concat(picked_ubs)



    def match_2hop_connected(self, maxdegree: int):
        """
        For a unmatched vertex/row, for all of its adjacent vertexes globally,
        create a new adjmat where for each 
        """
        row_flags = torch.logical_and(
            self.matched == MASKVALUE,
            # Count on raw adjmat, not on the masked adjmat.
            torch.count_nonzero(self.adj_lists != MASKVALUE, dim=1) < maxdegree
        )

        unmatched_vids = self.vertexes_ids[row_flags]
        unmatched_adjmat = self.adj_lists[row_flags, :]

        dist_env = get_runtime_dist_env()

        active_adj_vids_subs = []
        subadjmats = []
        for w in range(dist_env.world_size):
            if w == dist_env.rank:
                dist_env.broadcast_object_list(w, [unmatched_adjmat.shape])
                w_vids = dist_env.broadcast(w, unmatched_vids)
                w_adjmat = dist_env.broadcast(w, unmatched_adjmat)
            else:
                [(nv, width)] = dist_env.broadcast_object_list(w)
                w_vids = dist_env.broadcast(w, shape=(nv,), dtype=torch.int64)
                w_adjmat = dist_env.broadcast(w, shape=(nv, width), dtype=torch.int64)

            # inverse the rows and columns of w_adjmat, arrange a new partition
            # of adjmat in the order of `self.vertex_ids`
            isin_flags = torch.isin(w_adjmat, self.vertexes_ids)
            w_adj_vids = w_adjmat[isin_flags]

            # e.g. [[row1, col1], [row1, col2], [row2, col3], [row3, col4], ...]
            w_adjmat_coords = isin_flags.nonzero()

            w_vids_repeated = w_vids[w_adjmat_coords[:, 0]]

            assert w_adj_vids.shape[0] == w_vids_repeated.shape[0]

            # This is sort to natural number, not to the order of self.vertex_ids
            sorted_adj_vids, pos = w_adj_vids.sort()
            w_vids_reordered = w_vids_repeated[pos]
            
            unique_adj_vids, counts = torch.unique_consecutive(sorted_adj_vids, return_counts=True)
            max_count = int(torch.max(counts))

            n_active = unique_adj_vids.shape[0]

            flat_offsets = concat_aranges(counts) + \
                torch.repeat_interleave(torch.arange(n_active), counts)

            active = torch.full((n_active, max_count), fill_value=MASKVALUE, dtype=torch.int64)
            active.reshape(-1).scatter_(dim=0, index=flat_offsets, src=w_vids_reordered)

            column_wise_sub_adjmat = torch.full((self.vertexes_ids.shape[0], max_count), fill_value=MASKVALUE, dtype=torch.int64)
            column_wise_sub_adjmat[unique_adj_vids, :] = active

            active_adj_vids_subs.append(unique_adj_vids)
            subadjmats.append(column_wise_sub_adjmat)
        # end for w in world_size

        # TODO mask out new col-wise adjmat using matched
        # TODO decouple Matcher class to use these as new masked_adj_X for 2hop.
        column_wise_adjmat = torch.concat(subadjmats, dim=1)

        active_adj_vids = torch.concat(active_adj_vids_subs, dim=0).unique()
        row_flags = torch.isin(self.vertexes_ids, active_adj_vids)

        for w in range(dist_env.world_size):
            if w == dist_env.rank:
                self.core_2hop_connected(row_flags)

            self.sync_matched(w)


    def match_2hop_same_adj_list(self):
        """
        Match 2-hop vertexes only if their adjacent lists are exactly the same.
        We hash the adj list of unmatched vertexes, and distribute them among
        workers (in whatever order the hash method results in).
        """
        pass

def concat_aranges(sizes, dtype=torch.int64):
    cum_sizes = torch.cumsum(sizes, dim=0)

    n = int(sum(sizes))
    increases = torch.full((n,), fill_value=1, dtype=torch.int64)
    increases[0] = 0
    increases[cum_sizes[:-1]] = -sizes[:-1]

    return torch.cumsum(increases, dim=0)
    
def create_coaser_graph():
    """
    # (N,)  max() == CN
    cmap = torch.empty()

    distribute CN coarser nodes  among workers (CN/ws,) in natural order

    each worker has a part cmap for self.vertex_ids

    for adjlist, we need broadcasted cmap to map to cvid
    """
    dist_env = get_runtime_dist_env()

    cvids = range()  # coarser part on this worker

    for w in range(dist_env.world_size):
        vertex_ids = torch.empty()
        matched = torch.empty()
        

    cmap = torch.empty()
    adj_lists = torch.empty()
    adj_weights = torch.empty()

    cvids = cmap[vertex_ids]
    cmapped_adj_lists = torch.empty()


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
        end = min(self.per_worker_nv * (rank + 1), self.nv)
        return start, end
        


def coarsen(
    rowptr: torch.Tensor, colidx: torch.Tensor, adjwgt: torch.Tensor, dist_config: DistConfig,
    # max_v_weight: float
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

    start, end = dist_config.get_start_end()
    assert rowptr.shape[0] -1 == end - start
    
    # local vids to global vids
    matched = torch.full((end - start,), fill_value=MASKVALUE, dtype=torch.int64)
    masked_colidx = colidx.clone()

    for w in range(dist_env.world_size):
        if w == dist_env.rank:
            # int(?, 2): both values are global IDs, but the [:, 0] are held by this worker.
            matched_vid_pairs = torch.full((end - start, 2), fill_value=MASKVALUE, dtype=torch.int64)


            n_new_matches = print(matched, rowptr, masked_colidx, adjwgt, matched_vid_pairs)
            matched_vid_pairs = matched_vid_pairs[:n_new_matches, :]

            matched[matched_vid_pairs[:, 0] - start] = matched_vid_pairs[:, 1]

            # TODO is broadcast necessary which includes previous workers?
            #
            # Broadcast the whole result, to mask out adj list for each worker.
            dist_env.broadcast_object_list(w, [matched_vid_pairs.shape[0]])
            w_matched_pairs = dist_env.broadcast(w, matched_vid_pairs)

        else:
            [npairs] = dist_env.broadcast_object_list(w)
            w_matched_pairs = dist_env.broadcast(w, shape=(npairs, 2), dtype=torch.int64)

        # update matched records
        to_this_mask = (w_matched_pairs[:, 1] - start) >= 0
        to_this_pairs = w_matched_pairs[to_this_mask]
        matched[to_this_pairs[:, 1] - start] = to_this_pairs[:, 0]

        # mask out matched adj cells
        masked_colidx[torch.isin(masked_colidx, w_matched_pairs)] = MASKVALUE
        
    this_unmatched_n = (matched == -1).count_nonzero()
    # TODO will be self-match?
    this_matched_n = end - start - this_unmatched_n
    coarser_vid_map = torch.full((end - start,), fill_value=MASKVALUE, dtype=torch.int64)


@dataclass
class CoarseningLevel:
    dist_config: DistConfig




def part_kway(
    rowptr, colidx, adjwgt
):
    pass