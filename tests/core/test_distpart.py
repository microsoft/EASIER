# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from unittest.mock import patch
import torch
import pytest

import h5py
import tempfile
import os

import easier.cpp_extension as _C

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
            vec(0, 13, 14, 15, 20),      # v11 => 13
            vec(20, 21),         # v12  => 20
            vec(11, 20),         # v13  <= 11
            vec(11, 15, 20),     # v14  => 15
            vec(11, 14, 20),     # v15  <= 14
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
            -1, 13, 20, 11, 15, 14
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
