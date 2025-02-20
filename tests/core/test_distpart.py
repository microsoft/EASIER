# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from unittest.mock import patch
import torch
import pytest

import h5py
import tempfile
import os
import numpy
import scipy.sparse

from easier.core.distpart import CoarseningLevel, DistConfig, \
    gather_csr_graph, assign_cvids_unmatched, assign_cvids_colocated, \
    align_coarser_vids, get_csr_mask_by_rows, CoarseningRowDataExchanger, \
    exchange_cadj_adjw, merge_cadj_adjw, distpart_kway, uncoarsen_level, \
    metis_wrapper
from easier.core.runtime.dist_env import get_cpu_dist_env
import easier.cpp_extension as _C

from ..utils import mpirun_singlenode, assert_tensor_list_equal

def _C_hem(
    start: int, end: int, matched: torch.Tensor,
    rowptr: torch.Tensor, colidx: torch.Tensor, adjwgt: torch.Tensor
):
    # For hint and auto-completion only.
    _C.locally_match_heavy_edge(
        start, end, matched, rowptr, colidx, adjwgt
    )

def vec(*vs):
    return torch.tensor(vs, dtype=torch.int64)

class TestCppDistPart:
    def test_isolated_rows(self):
        colidxs = [
            vec(), # v10: isolated row -- => 11
            vec(5, 12, 22),  # v11   -- <= 10
            vec(7, 11, 21),  # v12
            vec() # v13: isolated row -- no more, unmatched
        ]

        start, end = 10, 10 + len(colidxs)
        rowptr = torch.tensor(
            [0] + [c.shape[0] for c in colidxs], dtype=torch.int64
        ).cumsum(dim=0)
        colidx = torch.concat(colidxs)
        matched = torch.full((end - start,), -1, dtype=torch.int64)

        _C_hem(
            start, end,
            matched,
            rowptr, colidx,
            adjwgt=torch.ones_like(colidx, dtype=torch.int64),
        )
        assert torch.equal(matched, vec(
            11, 10, 21, -1
        ))

    def test_subsequent_matching(self):
        colidxs = [
            vec(0, 1, 2),        # v10
            vec(0, 13, 14, 15, 16, 20),      # v11 => 13
            vec(20, 21),         # v12  => 20
            vec(11, 20),         # v13  <= 11
            vec(11, 15, 20),     # v14  => 15
            vec(11, 14, 20),     # v15  <= 14
            vec(11),             # -1
        ]

        start, end = 10, 10 + len(colidxs)
        rowptr = torch.tensor(
            [0] + [c.shape[0] for c in colidxs], dtype=torch.int64
        ).cumsum(dim=0)
        colidx = torch.concat(colidxs)
        matched = torch.full((end - start,), -1, dtype=torch.int64)
        _C_hem(
            start, end,
            matched,
            rowptr, colidx,
            adjwgt=torch.ones_like(colidx, dtype=torch.int64),
        )
        assert torch.equal(matched, vec(
            -1, 13, 20, 11, 15, 14, -1
        ))


    def test_match_heavy_edge(self):
        colidxs = [
            vec(    11, 12, 13, 14),
            vec(10,     12, 13, 14),
            vec(10, 11,     13, 14),
            vec(10, 11, 12,     14),
            vec(10, 11, 12, 13    )
        ]
        adjwgts = [
            vec(    1,  2,  3,  4),  # => 14
            vec(1,      5,  6,  7),  # => 13
            vec(2,  5,      8,  9),  # unmatched
            vec(3,  6,  8,     10),  # <= 11
            vec(4,  7,  9, 10,   ),  # <= 10
        ]

        start, end = 10, 10 + len(colidxs)
        rowptr = torch.tensor(
            [0] + [c.shape[0] for c in colidxs], dtype=torch.int64
        ).cumsum(dim=0)
        colidx = torch.concat(colidxs)
        adjwgt = torch.concat(adjwgts)
        matched = torch.full((end - start,), -1, dtype=torch.int64)
        _C_hem(
            start, end,
            matched,
            rowptr, colidx,
            adjwgt,
        )
        assert torch.equal(matched, vec(
            14, 13, -1, 11, 10
        ))

    def test_adj_already_matched(self):
        colidxs = [
            vec(14),        # v10 => 14
            vec(15),        # v11 => 15
            vec(14, 15),    # unmatched
            vec(),          # unmatched
            vec(10, 12),    # v14  => 10
            vec(11, 12),    # v15  <= 11
        ]

        start, end = 10, 10 + len(colidxs)
        rowptr = torch.tensor(
            [0] + [c.shape[0] for c in colidxs], dtype=torch.int64
        ).cumsum(dim=0)
        colidx = torch.concat(colidxs)
        matched = torch.full((end - start,), -1, dtype=torch.int64)
        _C_hem(
            start, end,
            matched,
            rowptr, colidx,
            adjwgt=torch.ones_like(colidx, dtype=torch.int64),
        )
        assert torch.equal(matched, vec(
            14, 15, -1, -1, 10, 11
        ))


def worker__test_gather_csr_rowptr(local_rank: int, world_size: int):
    from easier.core.runtime.dist_env import set_runtime_dist_env_backend
    set_runtime_dist_env_backend('cpu')

    colidx = vec(1, 2, 3, 4, 5, 6)
    rowptr = vec(0, 3, 6)
    csr = gather_csr_graph(
        0,
        CoarseningLevel(
            DistConfig(2 * world_size, [2] * world_size),
            torch.full((2,), fill_value=local_rank, dtype=torch.int64),
            rowptr,
            colidx,
            colidx + 10
        )
    )
    if local_rank == 0:
        assert csr is not None
        vwgt0, rowptr0, colidx0, adjw0 = csr
        assert torch.equal(vwgt0, torch.concat([
            vec(1, 1) * i for i in range(world_size)
        ]))
        assert torch.equal(rowptr0, torch.concat([vec(0)] + [
            vec(3, 6) + vec(6, 6) * i for i in range(world_size)
        ]))
        assert torch.equal(colidx0, torch.concat([colidx] * world_size))
        assert torch.equal(adjw0, torch.concat([colidx + 10] * world_size))

def test_gather_csr_rowptr():
    mpirun_singlenode(4, worker__test_gather_csr_rowptr)


class TestCoarserVertexID:
    def test_unmatched_full(self):
        matched = torch.full((10,), -1, dtype=torch.int64)
        cvids = torch.full((10,), -1, dtype=torch.int64)
        unmatched_n = assign_cvids_unmatched(matched, cvids, 33)
        assert unmatched_n == 10
        assert torch.all(matched == -1)
        assert torch.equal(cvids, torch.arange(33, 43))
        
    def test_unmatched_some(self):
        matched = torch.arange(10)
        matched[1:10:3] = -1
        cvids = torch.full((10,), -1, dtype=torch.int64)
        unmatched_n = assign_cvids_unmatched(matched, cvids, 33)
        assert unmatched_n == 3

        expected_cvids = torch.full((10,), -1, dtype=torch.int64)
        expected_cvids[1] = 33
        expected_cvids[4] = 34
        expected_cvids[7] = 35
        assert torch.equal(cvids, expected_cvids)
        
    def test_colocated(self):
        matched = torch.arange(40, 50)

        matched[1] = 15
        matched[5] = 11

        matched[3] = 19
        matched[9] = 13

        matched[6] = 17
        matched[7] = 16

        cvids = torch.full((10,), -1, dtype=torch.int64)
        colocated_n = assign_cvids_colocated(10, 20, matched, cvids, 33)
        assert colocated_n == 3

        expected_cvids = torch.full((10,), -1, dtype=torch.int64)
        expected_cvids[1] = expected_cvids[5] = 33
        expected_cvids[3] = expected_cvids[9] = 34
        expected_cvids[6] = expected_cvids[7] = 35
        assert torch.equal(cvids, expected_cvids)

    def test_align_remote(self):
        cvids = torch.full((10,), -1, dtype=torch.int64)
        align_coarser_vids(
            90, 95, vec(33, 44, 55, 66, 77),
            matched=vec(
                93, 92, 91, 94, 90, 91, 89, 95, 92, 94
            ),
            cvids=cvids
        )
        assert torch.equal(
            cvids,
            vec(
                66, 55, 44, 77, 33, 44, -1, -1, 55, 77
            )
        )


def test_csr_mask():
    colidxs = [
        torch.arange(i + 1) * 3 + i + 1
        for i in range(8)
    ]
    #   1
    #   2   5
    #   3   6   9
    #   4   7  10  13   
    #   ...

    rowptr = torch.tensor(
        [0] + [c.shape[0] for c in colidxs], dtype=torch.int64
    ).cumsum(dim=0)
    colidx = torch.concat(colidxs)
    nnz = colidx.shape[0]


    full_mask = get_csr_mask_by_rows(
        rowptr,
        torch.ones((8,), dtype=torch.bool),
        nnz
    )
    assert full_mask.shape[0] == nnz
    assert torch.all(full_mask == True)

    none_mask = get_csr_mask_by_rows(
        rowptr,
        torch.zeros((8,), dtype=torch.bool),
        nnz
    )
    assert none_mask.shape[0] == nnz
    assert torch.all(none_mask == False)

    even_mask = get_csr_mask_by_rows(
        rowptr,
        vec(1, 0, 1, 0, 1, 0, 1, 0) > 0,
        nnz
    )
    assert even_mask.shape[0] == nnz
    assert torch.equal(
        even_mask,
        torch.concat([
            (torch.ones if i % 2 == 0 else torch.zeros)(
                (i + 1,), dtype=torch.bool
            ) for i in range(8)
        ])
    )

    odd_mask = get_csr_mask_by_rows(
        rowptr,
        vec(0, 1, 0, 1, 0, 1, 0, 1) > 0,
        nnz
    )
    assert odd_mask.shape[0] == nnz
    assert torch.equal(
        odd_mask,
        torch.concat([
            (torch.ones if i % 2 == 1 else torch.zeros)(
                (i + 1,), dtype=torch.bool
            ) for i in range(8)
        ])
    )

    island_mask = get_csr_mask_by_rows(
        rowptr,
        vec(1, 1, 0, 0, 1, 1, 1, 0) > 0,
        nnz
    )
    assert island_mask.shape[0] == nnz
    assert torch.equal(
        island_mask,
        torch.concat([
            torch.full((1,), 1),
            torch.full((2,), 1),
            torch.full((3,), 0),
            torch.full((4,), 0),
            torch.full((5,), 1),
            torch.full((6,), 1),
            torch.full((7,), 1),
            torch.full((8,), 0),
        ]) > 0
    )

def test_csr_mask_with_empty_rows():
    colidxs = [
        vec(),
        vec(),
        vec(1, 2, 3),
        vec(),
        vec(1, 2, 3),
        vec(1, 2, 3),
        vec(),
        vec(1, 2, 3),
    ]
    rowptr = torch.tensor(
        [0] + [c.shape[0] for c in colidxs], dtype=torch.int64
    ).cumsum(dim=0)
    colidx = torch.concat(colidxs)
    nnz = colidx.shape[0]

    full_mask = get_csr_mask_by_rows(
        rowptr,
        torch.ones((8,), dtype=torch.bool),
        nnz
    )
    assert full_mask.shape[0] == nnz
    assert torch.all(full_mask == True)

    none_mask = get_csr_mask_by_rows(
        rowptr,
        torch.zeros((8,), dtype=torch.bool),
        nnz
    )
    assert none_mask.shape[0] == nnz
    assert torch.all(none_mask == False)

    even_mask = get_csr_mask_by_rows(
        rowptr,
        vec(1, 0, 1, 0, 1, 0, 1, 0) > 0,
        nnz
    )
    assert even_mask.shape[0] == nnz
    assert torch.equal(
        even_mask,
        vec(
            # T
            # F
            1, 1, 1,  # T
            # F
            1, 1, 1,  # T
            0, 0, 0,  # F
            # T
            0, 0, 0,  # F
        ) > 0
    )

    odd_mask = get_csr_mask_by_rows(
        rowptr,
        vec(0, 1, 0, 1, 0, 1, 0, 1) > 0,
        nnz
    )
    assert odd_mask.shape[0] == nnz
    assert torch.equal(
        odd_mask,
        vec(
            # F
            # T
            0, 0, 0,  # F
            # T
            0, 0, 0,  # F
            1, 1, 1,  # T
            # F
            1, 1, 1,  # T
        ) > 0
    )

    island_mask = get_csr_mask_by_rows(
        rowptr,
        vec(1, 1, 0, 0, 1, 1, 1, 0) > 0,
        nnz
    )
    assert island_mask.shape[0] == nnz
    assert torch.equal(
        island_mask,
        vec(
            # T
            # T
            0, 0, 0,  # F
            # F
            1, 1, 1,  # T
            1, 1, 1,  # T
            # T
            0, 0, 0,  # F
        ) > 0
    )


def worker__test_row_exchanger(local_rank: int, world_size: int):
    assert world_size == 3
    # 0, 1, ..., 9
    # 0, 1, ..., 12
    # 0, 1, ..., 15
    cvids = torch.arange(10 + local_rank * 3)
    cnv = 20

    # [0, 5)
    # [5, 13)
    # [13, 20)
    c_dist_config = DistConfig(cnv, [5, 8, 7])

    xchg = CoarseningRowDataExchanger(c_dist_config, cvids)

    if local_rank == 0:
        assert_tensor_list_equal(xchg.rows_to_other_masks, [
            torch.arange(10) < 5,
            torch.logical_and(5 <= torch.arange(10), torch.arange(10) < 5 + 8),
            torch.zeros((10,), dtype=torch.bool)
        ])
        assert torch.equal(xchg.cvids_unmerged_on_this, torch.concat([
            torch.arange(5),
            torch.arange(5),
            torch.arange(5)
        ]))
    elif local_rank == 1:
        assert_tensor_list_equal(xchg.rows_to_other_masks, [
            torch.arange(13) < 5,
            torch.logical_and(5 <= torch.arange(13), torch.arange(13) < 5 + 8),
            torch.zeros((13,), dtype=torch.bool)
        ])
        assert torch.equal(xchg.cvids_unmerged_on_this, torch.concat([
            torch.arange(5, 10),
            torch.arange(5, 13),
            torch.arange(5, 13)
        ]))
    elif local_rank == 2:
        assert_tensor_list_equal(xchg.rows_to_other_masks, [
            torch.arange(16) < 5,
            torch.logical_and(5 <= torch.arange(16), torch.arange(16) < 5 + 8),
            5 + 8 <= torch.arange(16),
        ])
        assert torch.equal(xchg.cvids_unmerged_on_this, torch.concat([
            vec(),
            vec(),
            torch.arange(13, 16)
        ]))

def test_row_exchanger():
    mpirun_singlenode(3, worker__test_row_exchanger)


def worker__test_exchange_merge_adj(local_rank: int, world_size: int):
    """
    Exchange tests only. Not to map cvids, we can use whatever adj values here.
    """
    assert world_size == 3
    adjs = [
        vec(3,   5,   7,   9),
        vec(11,  15,  19),
        vec(22,  33,  41,  42,  43),
        vec(31, 35, 39, 43)
    ]

    rowptr = torch.tensor(
        [0] + [c.shape[0] for c in adjs], dtype=torch.int64
    ).cumsum(dim=0)
    local_adj = torch.concat(adjs) + local_rank

    nv = 12
    cnv = 48

    cvids = [
    # len   4  3  5   4
        vec(7, 7, 22, 33),      # 1 1 2 3 -- c_rank
        vec(7, 15, 20, 33),     # 1 2 2 3
        vec(15, 17, 18, 20)     # 2 2 2 2
    ][local_rank]

    # [0, 10)
    # [10, 32)
    # [32, 48)
    c_dist_config = DistConfig(cnv, [10, 22, 16])

    xchg = CoarseningRowDataExchanger(c_dist_config, cvids)

    sizes, adj1, adj2 = exchange_cadj_adjw(xchg, rowptr, local_adj, -local_adj)

    assert torch.equal(adj1, -adj2)

    if local_rank == 0:
        assert torch.equal(xchg.cvids_unmerged_on_this, torch.concat([
            vec(7, 7),
            vec(7),
            vec(),
        ]))
        assert torch.equal(sizes, vec(4, 3, 4))
        assert torch.equal(adj1, torch.concat([
            vec(3, 5, 7, 9),
            vec(11, 15, 19),
            1 + vec(3, 5, 7, 9),  # 4 6 8 10
        ]))
    elif local_rank == 1:
        assert torch.equal(xchg.cvids_unmerged_on_this, torch.concat([
            vec(22),
            vec(15, 20),
            vec(15, 17, 18, 20),
        ]))
        assert torch.equal(sizes, vec(5, 3, 5, 4, 3, 5, 4))
        expect_adj1_r1 = torch.concat([
            vec(22,  33,  41,  42,  43),  # => 22
            1 + vec(11,  15,  19),  # 12 16 20 => 15
            1 + vec(22,  33,  41,  42,  43),  # 23 34 42 43 44 => 20
            2 + vec(3,   5,   7,   9),  # 5 7 9 11 => 15
            2 + vec(11,  15,  19),  # 13 17 21 => 17
            2 + vec(22,  33,  41,  42,  43),  # 24 35 43 44 45 => 18
            2 + vec(31, 35, 39, 43)  # 33 37 41 45 => 20
        ])
        assert torch.equal(adj1, expect_adj1_r1)
    elif local_rank == 2:
        assert torch.equal(xchg.cvids_unmerged_on_this, torch.concat([
            vec(33),
            vec(33),
            vec(),
        ]))
        assert torch.equal(sizes, vec(4, 4))
        assert torch.equal(adj1, torch.concat([
            vec(31, 35, 39, 43),
            1 + vec(31, 35, 39, 43),  # 32 36 40 44
        ]))

    crowptr, ccolidx, cw = merge_cadj_adjw(xchg, sizes, adj1, adj2)

    if local_rank == 0:
        assert torch.equal(crowptr, vec(
            0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 10
        ))
        assert torch.equal(ccolidx, vec(
            3, 4, 5, 6, 8, 9, 10, 11, 15, 19  # no 7 -- self
        ))
        assert torch.equal(cw, -vec(
            3, 4, 5, 6, 8, 9, 10, 11, 15, 19
        ))
    elif local_rank == 1:
        assert torch.equal(ccolidx, vec(
            # 15:
            5, 7, 9, 11, 12, 16, 20,
            # 17:
            13, 21,  # rm 17
            # 18:
            24, 35, 43, 44, 45,
            # 20:
            23, 33, 34, 37, 41, 42, 43, 44, 45,
            # 22:
            33, 41, 42, 43  # em 22
        ))

def test_exchange_merge_adj():
    mpirun_singlenode(3, worker__test_exchange_merge_adj)

def worker__test_preserve_symmetry(local_rank, world_size):
    """
    Other tests may not maintain the symmetry of the input CSR
    which is expected during the coarsening.
    """
    nv = 100
    ne = 2000
    dist_env = get_cpu_dist_env()

    dist_config = DistConfig.create_default(nv)

    if local_rank == 0:
        torch.manual_seed(2345)
        x = torch.randint(0, nv, (ne,))
        y = torch.randint(0, nv, (ne,))
        w = torch.ones((ne,), dtype=torch.int64)

        g = scipy.sparse.csr_matrix(
            (w, (x, y)), shape=(nv, nv), dtype=numpy.int64
        )
        g = g + g.transpose()
        g = g.tolil()
        g.setdiag(0)
        g = g.tocsr()

        rs = []
        cs = []
        ws = []
        for i in range(world_size):
            start, end = dist_config.get_start_end(i)
            subg = g[start: end]

            rs.append(torch.from_numpy(subg.indptr).to(torch.int64))
            cs.append(torch.from_numpy(subg.indices).to(torch.int64))
            ws.append(torch.from_numpy(subg.data).to(torch.int64))
            
    else:
        rs, cs, ws = None, None, None

    rowptr = dist_env.scatter_object(0, rs)  # type: ignore
    colidx = dist_env.scatter_object(0, cs)  # type: ignore
    adjw = dist_env.scatter_object(0, ws)  # type: ignore

    def _hook_metis(nparts, rowptr, colidx, vwgt, adjwgt):
        cnv = int(rowptr.shape[0]) - 1

        assert torch.all(vwgt >= 1)
        assert torch.any(vwgt >= 2)

        assert torch.all(adjwgt >= 1)
        assert torch.any(adjwgt >= 2)

        g = scipy.sparse.csr_matrix(
            (adjwgt, colidx, rowptr), shape=(cnv, cnv)
        ).todense()
        assert numpy.all(g - g.T == 0)

        # fake result
        r = torch.zeros((cnv,), dtype=torch.int64)
        r[:world_size] = torch.arange(world_size)
        return 99, r

    # Because of the `while True` loop, we have at least coarsened once.
    with patch(
        f'{metis_wrapper.__module__}.{metis_wrapper.__name__}',
        wraps=_hook_metis
    ) as mock:
        membership = distpart_kway(dist_config, rowptr, colidx, adjw)

    if local_rank == 0:
        mock.assert_called()


def test_preserve_symmetry():
    mpirun_singlenode(3, worker__test_preserve_symmetry)


def worker__test_uncoarsen_level(local_rank: int, world_size: int):
    assert world_size == 3
    c_dist_config = DistConfig(15, [5, 5, 5])
    c_mb = torch.full((5,), fill_value=2 - local_rank, dtype=torch.int64)

    cvids = torch.arange(5 * local_rank, 5 * (local_rank + 1))
    mb = uncoarsen_level(c_dist_config, c_mb, cvids)
    assert torch.equal(mb, c_mb)

    cvids = 14 - torch.arange(5 * local_rank, 5 * (local_rank + 1))
    mb = uncoarsen_level(c_dist_config, c_mb, cvids)
    assert torch.equal(mb, 2 - c_mb)

    cvids = vec(2, 4, 6, 8, 10)
    mb = uncoarsen_level(c_dist_config, c_mb, cvids)
    assert torch.equal(mb, vec(2, 2, 1, 1, 0))

    cvids = vec(2, 4, 6, 8, 10) + local_rank
    mb = uncoarsen_level(c_dist_config, c_mb, cvids)
    assert torch.equal(mb, [
        vec(2, 2, 1, 1, 0),
        vec(2, 1, 1, 1, 0),
        vec(2, 1, 1, 0, 0)
    ][local_rank])

def test_uncoarsen_level():
    mpirun_singlenode(3, worker__test_uncoarsen_level)