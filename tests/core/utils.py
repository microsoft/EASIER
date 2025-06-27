# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Tuple
from unittest.mock import patch
import contextlib
import torch

from easier.core.passes.tensor_grouping import EasierTensorGroup
from easier.core.runtime.data_loader import \
    _get_offset_exactly_nparts
from easier.core.distpart import \
    metis_wrapper as _metis
from easier.core.passes.tensor_group_partition import ElemPart, \
    distpart_kway as _kway, \
    synchronize_partition_result as _sync_elempart
from easier.core.runtime.dist_env import get_runtime_dist_env


@contextlib.contextmanager
def multi_stage_zero_length_partition(
    cross_grp_defs: tuple  # Tuple[S|R|T, S|R|T]
):
    """
    In a distributed setting, and on these ahead-of-time compilation stages,
    inject zero-length partitions:

    1.  In tensor_group_partition and when preparing the distributed adjmat:

        `get_selector_reducer_idx_partition` is called to get a idx part and
        the range of that part in idx;

        This case stands for a very small but distributed dataset.

        TODO the local part of the adjmat is not injected by this patch,
        because the adjmat is a concat of all TensorGroups and is unlikely to
        be smaller than world size.

    2.  In distpart, the rank-0 METIS returns with one rank having no vertex
        assignment:

        `metis_wrapper` is called;
        TODO in the future this initial assignment may get refined during
        uncoarsening

        This case stands for e.g. boundary conditions and METIS happens to
        partition into a "#" shape rather than a "*" shape.

    3.  In distpart, the overall k-way partition returns with one rank having
        no vertex assignment (this is after uncoarsening)

        `distpart_kway` is called;

        Additional to #2.

    4.  After tensor_group_partition, inject zero ElemParts, and in a
        cross pattern:
        elemparts[grpI][rankJ] = 0, and elemparts[grpI+1][rankJ+1] = 0,
        so that in sparse_encoding we get both sides of both Selector/Reducer
        subprocedures become zero-length.

    Each kind of results above plays as the boundary between stages, so it's
    OK to inject into each of them.
    """
    dist_env = get_runtime_dist_env()
    rank = dist_env.rank
    world_size = dist_env.world_size

    def _get_dt_nparts_0len(orig_len: int, nparts: int, part: int):
        # last worker has no assignment
        per_worker_len = orig_len // nparts
        start = per_worker_len * part

        if part + 1 == nparts:
            start = orig_len
            end = orig_len
        elif part + 2 == nparts:
            end = orig_len
        else:
            end = per_worker_len * (part + 1)

        return start, end

    def _metis_wrapper_0len(*args, **kwargs) -> Tuple[int, torch.Tensor]:
        # only run on rank-0
        # before uncoarsening (and refinement)
        # last worker has no assignment
        ncuts, membership = _metis(*args, **kwargs)
        ncuts = ncuts + 999
        membership[membership == 0] = 1
        return ncuts, membership

    def _kway_0len(*args, **kwargs):
        # each worker has a part of membership tensor
        # after uncoarsening (and refinement)
        # worker-0 has no assignment
        local_membership = _kway(*args, **kwargs)
        local_membership[local_membership == 0] = 1
        return local_membership

    def _sync_elempart_0len(
        tensor_groups,
        local_membership: torch.Tensor  # many 0s, polluted by prev patches
    ):
        # grp1 selected to grp2, grp2 reduced to grp1
        grpdef1, grpdef2 = cross_grp_defs
        grp1 = grpdef1.easier_tensor_group
        grp2 = grpdef2.easier_tensor_group
        assert isinstance(grp1, EasierTensorGroup)
        assert isinstance(grp2, EasierTensorGroup)

        elemparts = _sync_elempart(tensor_groups, local_membership)
        for grp, ep in list(elemparts.items()):
            #       grp1    grp2
            # rank0  X       X  # removed, aligns with kway deselects rank-0
            # rank1  +       +
            # rank2  X       O  # grp1 moved to rank1
            # rank3  O       X  # grp2 moved to rank1
            # ...    O       O
            if grp is grp1:
                src_ranks = [0, 2]
            elif grp is grp2:
                src_ranks = [0, 3]
            else:
                continue

            assert world_size >= 2, "at least 2 workers"
            dst_rank = 1

            empty_ep = torch.empty(
                (0,), dtype=ep.idx.dtype, device=ep.idx.device
            )
            ep_idxes = dist_env.gather_object_list(0, ep.idx)

            if dist_env.rank == 0:
                assert ep_idxes is not None
                if world_size < 4:
                    # in case the number of UT distributed subprocesses is
                    # less than 4.
                    ep_idxes = ep_idxes + [empty_ep] * (4 - world_size)

                dst_ep_idx = torch.concat([
                    ep_idxes[src_ranks[0]],
                    ep_idxes[src_ranks[1]],
                    ep_idxes[dst_rank]
                ])

                ep_idxes[src_ranks[0]] = ep_idxes[src_ranks[1]] = empty_ep
                ep_idxes[dst_rank] = dst_ep_idx

                lengths = [p.shape[0] for p in ep_idxes][:world_size]

                ep_idx = dist_env.scatter_object_list(0, ep_idxes[:world_size])
                lengths = dist_env.broadcast_object_list(0, lengths)
            else:
                ep_idx = dist_env.scatter_object_list(0)
                lengths = dist_env.broadcast_object_list(0)

            elemparts[grp] = ElemPart(None, ep_idx, lengths, ep.hint)

        return elemparts

    dataloader_module = _get_offset_exactly_nparts.__module__
    distpart_module = _metis.__module__
    tensor_partition_module = _sync_elempart.__module__

    with patch(
        f'{dataloader_module}.{_get_offset_exactly_nparts.__name__}',
    ) as dt_mock, \
        patch(
        f'{distpart_module}.{_metis.__name__}',
    ) as metis_mock, \
        patch(
        f'{tensor_partition_module}.{_kway.__name__}',
    ) as kway_mock, \
        patch(
        f'{tensor_partition_module}.{_sync_elempart.__name__}',
    ) as sync_ep_mock:
        dt_mock.side_effect = _get_dt_nparts_0len
        metis_mock.side_effect = _metis_wrapper_0len
        kway_mock.side_effect = _kway_0len
        sync_ep_mock.side_effect = _sync_elempart_0len

        yield

        dt_mock.assert_called()
        if rank == 0:
            metis_mock.assert_called_once()
        kway_mock.assert_called_once()
        sync_ep_mock.assert_called_once()
