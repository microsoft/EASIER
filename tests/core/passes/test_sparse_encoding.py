# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List, Union
from types import MethodType
import pytest
from unittest.mock import MagicMock, Mock, patch

import torch
from easier.core.passes.tensor_grouping import EasierTensorGroup
from easier.core.passes.tensor_group_partition import ElemPart as _EP_raw
import easier.core.runtime.dist_env as _JitRuntimeDistEnv
from easier.core.runtime.dist_env import DistEnv
from easier.core.module import Selector, Reducer
from easier.core.passes.utils import OrderedSet
from easier.core.passes.sparse_encoding.reorder_plan import \
    ReorderGraphBuilder, build_cascade_reorder_plan_on_rank0, \
    CascadeReorderStep
from easier.core.passes.sparse_encoding.sparse_encoding import \
    reorder_output_by_selector, rewrite_selector_instance, \
    reorder_input_by_reducer, rewrite_reducer_instance
from tests.utils import assert_tensor_list_equal, torchrun_singlenode


def vec(*longs):
    return torch.LongTensor(longs)


def ElemPart(idx, lengths):
    return _EP_raw(idx, lengths, 'NOHINT')


def test_break_reducer_cycle():
    g1 = EasierTensorGroup(OrderedSet(), 11, 'g1')
    g2 = EasierTensorGroup(OrderedSet(), 22, 'g2')
    g3 = EasierTensorGroup(OrderedSet(), 33, 'g3')
    r5 = Reducer(torch.arange(5), 99)
    r6 = Reducer(torch.arange(6), 99)
    r4 = Reducer(torch.arange(4), 99)
    builder = ReorderGraphBuilder([], [])
    builder.reducer_nnodes = {
        r5: 1,
        r6: 1,
        r4: 1,
    }
    builder.reducer_edges = {
        # Reducer.n does not matter as we are not running metadata propagation
        (g1, g2): r5,
        (g2, g3): r6,
        (g3, g1): r4  # break this
    }

    # g1->g2->g3-x->
    plan = build_cascade_reorder_plan_on_rank0(builder)

    assert plan == [
        CascadeReorderStep(g2, r5, g1),
        CascadeReorderStep(g3, r6, g2),
    ]


def test_break_selector_reducer_cycle():
    g1 = EasierTensorGroup(OrderedSet(), 11, 'g1')
    g2 = EasierTensorGroup(OrderedSet(), 22, 'g2')
    g3 = EasierTensorGroup(OrderedSet(), 33, 'g3')
    r5 = Reducer(torch.arange(5), 99)
    s9 = Selector(torch.arange(9))
    s8 = Selector(torch.arange(8))
    s7 = Selector(torch.arange(7))
    builder = ReorderGraphBuilder([], [])
    builder.reducer_edges = {
        # Reducer.n does not matter as we are not running metadata propagation
        (g1, g2): r5,
    }
    builder.reducer_nnodes = {
        r5: 1
    }
    builder.selector_edges = {
        (g2, g3): OrderedSet([
            s9,
            s7
        ]),
        (g3, g1): OrderedSet([
            s8
        ])
    }
    builder.selector_nnodes = {
        s7: 22,  # win
        s9: 11,  # lose conflict resolution to s7
        s8: 1    # lost cycle breaking to s7
    }

    # g1-R->g2-S7->g3-xS8x->
    plan = build_cascade_reorder_plan_on_rank0(builder)

    assert plan == [
        CascadeReorderStep(g2, r5, g1),
        CascadeReorderStep(g3, s7, g2),
    ]


def test_resolve_conflict():
    g1 = EasierTensorGroup(OrderedSet(), 11, 'g1')
    g2 = EasierTensorGroup(OrderedSet(), 22, 'g2')
    g3 = EasierTensorGroup(OrderedSet(), 33, 'g3')
    r5 = Reducer(torch.arange(5), 99)
    s22 = Selector(torch.arange(22))
    s33A = Selector(torch.arange(33))
    s33B = Selector(torch.arange(33))
    builder = ReorderGraphBuilder([], [])
    builder.reducer_edges = {
        # Reducer.n does not matter as we are not running metadata propagation
        (g1, g2): r5,
    }
    builder.reducer_nnodes = {
        r5: 1
    }
    builder.selector_edges = {
        (g1, g2): OrderedSet([
            s22  # lose to r5
        ]),

        (g1, g3): OrderedSet([
            s33A,  # lost to s33B
        ]),
        (g2, g3): OrderedSet([
            s33B,
        ])
    }
    builder.selector_nnodes = {
        s22: 99,
        s33A: 5,
        s33B: 6
    }

    plan = build_cascade_reorder_plan_on_rank0(builder)

    assert plan == [
        CascadeReorderStep(g2, r5, g1),
        CascadeReorderStep(g3, s33B, g2),
    ]


def worker__test_reroder_rewrite_selector(
    local_rank: int, world_size: int
):
    idx = vec(
        2, 14, 2, 14, 9, 16,
        14, 6, 2, 11, 14, 9,
        14, 9, 13, 7, 14, 4
    )
    selector = Selector(idx)

    input_elempart_idxes = [
        vec(0, 1, 4, 3, 2, 5),
        vec(6, 9, 8, 7, 10, 11),
        vec(12, 14, 13, 15, 16, 17)
    ]
    input_elempart = ElemPart(input_elempart_idxes[local_rank], [6, 6, 6])

    output_elempart_idxes = torch.arange(18).split(6)
    output_elempart = ElemPart(output_elempart_idxes[local_rank], [6, 6, 6])

    (input_gidx_to_this, output_gidx_on_this), \
        reordered_output_gidx_on_this = reorder_output_by_selector(
            selector, input_elempart, output_elempart)

    _s_out_gidx, _pos = output_gidx_on_this.sort()
    _s_in_gidx = input_gidx_to_this[_pos]

    assert torch.equal(_s_out_gidx, output_elempart_idxes[local_rank])

    if local_rank == 0:
        assert torch.equal(_s_in_gidx, vec(2, 14, 2, 14, 9, 16))
        assert torch.equal(reordered_output_gidx_on_this,
                           vec(0, 2, 4, 1, 3, 5))
    elif local_rank == 1:
        assert torch.equal(_s_in_gidx, vec(14, 6, 2, 11, 14, 9))
        assert torch.equal(reordered_output_gidx_on_this,
                           vec(8, 7, 11, 9, 6, 10))
    elif local_rank == 2:
        assert torch.equal(_s_in_gidx, vec(14, 9, 13, 7, 14, 4))
        assert torch.equal(reordered_output_gidx_on_this,
                           vec(17, 13, 15, 12, 16, 14))

    reordered_output_elempart = ElemPart(reordered_output_gidx_on_this,
                                         [6, 6, 6])
    rewrite_selector_instance(
        selector, input_gidx_to_this, output_gidx_on_this,
        input_elempart, reordered_output_elempart)

    if local_rank == 0:
        assert torch.equal(selector.idx, vec(4, 4, 6, 7, 7, 8))
        assert_tensor_list_equal(selector.runtime_halos_local_idxes, [
            vec(4),
            vec(4),
            vec(2)
        ])
    elif local_rank == 1:
        assert torch.equal(selector.idx, vec(0, 1, 2, 6, 7, 7))
        assert_tensor_list_equal(selector.runtime_halos_local_idxes, [
            vec(1),
            vec(0, 1, 5),
            vec(1, 3)
        ])
    elif local_rank == 2:
        assert torch.equal(selector.idx, vec(0, 1, 2, 4, 4, 5))
        assert_tensor_list_equal(selector.runtime_halos_local_idxes, [
            vec(1, 4),
            vec(1),
            vec(1, 2)
        ])


def test_reroder_rewrite_selector():
    torchrun_singlenode(3, worker__test_reroder_rewrite_selector)


def worker__test_reroder_rewrite_reducer(
    local_rank: int, world_size: int
):
    idx = vec(
        2, 14, 2, 14, 9, 16,
        14, 6, 2, 11, 14, 9,
        14, 9, 13, 7, 14, 4
    )
    reducer = Reducer(idx, 18)

    input_elempart_idxes = torch.arange(18).split(6)
    input_elempart = ElemPart(input_elempart_idxes[local_rank], [6, 6, 6])

    output_elempart_idxes = [
        vec(0, 1, 4, 3, 2, 5),
        vec(6, 9, 8, 7, 10, 11),
        vec(12, 14, 13, 15, 16, 17)
    ]
    output_elempart = ElemPart(output_elempart_idxes[local_rank], [6, 6, 6])

    (input_gidx_to_this, output_gidx_on_this), \
        reordered_input_elempart_idx = reorder_input_by_reducer(
            reducer, input_elempart, output_elempart)

    _s_in_gidx, _pos = input_gidx_to_this.sort()
    _s_out_gidx = output_gidx_on_this[_pos]

    if local_rank == 0:
        assert torch.equal(_s_in_gidx, vec(0, 2, 8, 17))
        assert torch.equal(_s_out_gidx, vec(2, 2, 2, 4))
        assert torch.equal(reordered_input_elempart_idx, vec(0, 2, 4, 1, 3, 5))
    elif local_rank == 1:
        assert torch.equal(_s_in_gidx, vec(4, 7, 9, 11, 13, 15))
        assert torch.equal(_s_out_gidx, vec(9, 6, 11, 9, 9, 7))
        assert torch.equal(reordered_input_elempart_idx,
                           vec(8, 7, 11, 9, 6, 10))
    elif local_rank == 2:
        assert torch.equal(_s_in_gidx, vec(1, 3, 5, 6, 10, 12, 14, 16))
        assert torch.equal(_s_out_gidx, vec(14, 14, 16, 14, 14, 14, 13, 14))
        assert torch.equal(reordered_input_elempart_idx,
                           vec(17, 13, 15, 12, 16, 14))

    reordered_input_elempart = ElemPart(
        reordered_input_elempart_idx, [6, 6, 6])
    rewrite_reducer_instance(
        reducer, input_gidx_to_this, output_gidx_on_this,
        reordered_input_elempart, output_elempart)

    if local_rank == 0:
        assert torch.equal(
            reducer.easier_reordering_selector_idx,  # type: ignore
            vec(3, 0, 1, 2))
        assert torch.equal(reducer.idx, vec(2, 4, 4, 4))
        assert_tensor_list_equal(reducer.runtime_halos_local_idxes, [
            vec(0, 1),
            vec(2),
            vec(3, 4, 5)
        ])
    elif local_rank == 1:
        assert torch.equal(
            reducer.easier_reordering_selector_idx,  # type: ignore
            vec(1, 0, 2, 4, 5, 3))
        assert torch.equal(reducer.idx, vec(0, 1, 1, 1, 3, 5))
        assert_tensor_list_equal(reducer.runtime_halos_local_idxes, [
            vec(0),
            vec(1, 2, 3),
            vec(4, 5)
        ])
    elif local_rank == 2:
        assert torch.equal(
            reducer.easier_reordering_selector_idx,  # type: ignore
            vec(0, 1, 3, 4, 5, 6, 7, 2))
        assert torch.equal(reducer.idx, vec(1, 1, 1, 1, 1, 1, 2, 4))
        assert_tensor_list_equal(reducer.runtime_halos_local_idxes, [
            vec(0),
            vec(1, 2),
            vec(3, 4, 5)
        ])


def test_reroder_rewrite_reducer():
    torchrun_singlenode(3, worker__test_reroder_rewrite_reducer)
