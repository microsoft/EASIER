# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List, Union
from types import MethodType
import pytest
from unittest.mock import MagicMock, Mock, patch

import torch
from torch.fx.node import Node

from easier.core import passes
from easier.core.jit import EasierTracer
from easier.core.passes.tensor_grouping import EasierTensorGroup
from easier.core.passes.tensor_group_partition import ElemPart as _EP_raw
import easier.core.runtime.dist_env as _JitRuntimeDistEnv
from easier.core.runtime.dist_env import DistEnv
from easier.core.module import Selector, Reducer
from easier.core.passes.utils import FX, OrderedSet
from easier.core.passes.sparse_encoding.sparse_encoding import \
    reorder_output_by_selector, rewrite_selector_instance, \
    reorder_input_by_reducer, rewrite_reducer_instance
from easier.core.passes.dataflow_distribution import \
    HaloExchangerInserter, ReorderingSelectorInserter, HaloExchanger
from tests.utils import assert_tensor_list_equal, mpirun_singlenode
import easier


def vec(*longs):
    return torch.LongTensor(longs)


def ElemPart(idx, lengths):
    return _EP_raw(idx, lengths, 'NOHINT')


def worker__test_halo_exchanger_insertion_for_selector(
    local_rank: int, world_size: int
):
    idx = vec(0)  # doesn't matter
    selector = Selector(idx)

    input_elempart_idxes = torch.arange(18).split(6)
    input_elempart = ElemPart(input_elempart_idxes[local_rank], [6, 6, 6])

    output_elempart_idxes = torch.arange(18).split(6)
    output_elempart = ElemPart(output_elempart_idxes[local_rank], [6, 6, 6])

    if local_rank == 0:
        # Only sends
        input_gidx_to_this = vec(0, 4, 2, 3, 1, 5)
        output_gidx_on_this = torch.arange(6)
    elif local_rank == 1:
        # No comm
        input_gidx_to_this = torch.arange(6, 12).flip(0)
        output_gidx_on_this = torch.arange(6, 12)
    elif local_rank == 2:
        # Only recvs
        input_gidx_to_this = vec(12, 1, 14, 15, 4, 17)
        output_gidx_on_this = torch.arange(12, 18)

    rewrite_selector_instance(
        selector, input_gidx_to_this, output_gidx_on_this,
        input_elempart, output_elempart)

    if local_rank == 0:
        assert torch.equal(selector.idx, vec(0, 4, 2, 3, 1, 5))
        assert_tensor_list_equal(selector.runtime_halos_local_idxes, [
            torch.arange(6),
            vec(),
            vec(1, 4)
        ])
    elif local_rank == 1:
        assert torch.equal(selector.idx, torch.arange(6).flip(0))
        assert_tensor_list_equal(selector.runtime_halos_local_idxes, [
            vec(),
            torch.arange(6),
            vec()
        ])
    elif local_rank == 2:
        assert torch.equal(selector.idx, vec(2, 0, 4, 5, 1, 7))
        assert_tensor_list_equal(selector.runtime_halos_local_idxes, [
            vec(),
            vec(),
            vec(0, 2, 3, 5)
        ])

    class M(easier.Module):
        def __init__(self):
            super().__init__()
            self.selector = selector
            self.v = easier.Tensor(torch.ones(6), mode='partition')

        def forward(self):
            r = self.selector(self.v)
            self.v[:] = r

    m = M()
    graph = EasierTracer().trace(m)
    [jm], [graph] = passes.group_tensors([m], [graph])

    _JitRuntimeDistEnv.init_runtime_dist_env('cpu')
    elemparts = {m.v.easier_tensor_group: output_elempart}
    HaloExchangerInserter(
        [jm], [graph], elemparts,  # type: ignore
    ).run()

    call_haloxchg: Node
    haloxchg: HaloExchanger
    if local_rank == 0:
        getattr_v, call_haloxchg, call_selector, setitem_v, out = graph.nodes

        haloxchg = jm.get_submodule(call_haloxchg.target)  # type: ignore
        assert haloxchg.concat_buffer_length is None

    elif local_rank == 1:
        getattr_v, call_selector, setitem_v, out = graph.nodes

    elif local_rank == 2:
        getattr_v, call_haloxchg, call_selector, setitem_v, out = graph.nodes

        haloxchg = jm.get_submodule(call_haloxchg.target)  # type: ignore
        assert haloxchg.concat_buffer_length == 8

    assert getattr_v.op == FX.GET_ATTR
    assert call_selector.op == FX.CALL_MODULE
    assert setitem_v.op == FX.CALL_FUNCTION


def test_halo_exchanger_insertion_for_selector():
    mpirun_singlenode(3, worker__test_halo_exchanger_insertion_for_selector)


def worker__test_halo_exchanger_insertion_for_reducer(
    local_rank: int, world_size: int
):
    idx = torch.arange(6)  # doesn't matter
    reducer = Reducer(idx, 6)  # assume to be local Reducer to pass metaprop

    class M(easier.Module):
        def __init__(self):
            super().__init__()
            self.reducer = reducer
            self.v = easier.Tensor(torch.ones(6), mode='partition')

        def forward(self):
            r = self.reducer(self.v)

    m = M()
    graph = EasierTracer().trace(m)
    [jm], [graph] = passes.group_tensors([m], [graph])

    input_elempart_idxes = torch.arange(18).split(6)
    input_elempart = ElemPart(input_elempart_idxes[local_rank], [6, 6, 6])

    output_elempart_idxes = torch.arange(18).split(6)
    output_elempart = ElemPart(output_elempart_idxes[local_rank], [6, 6, 6])

    if local_rank == 0:
        # Only sends, no reordering
        input_gidx_to_this = vec(0, 2, 3, 5)
        output_gidx_on_this = vec(2, 2, 3, 3)
    elif local_rank == 1:
        # No comm, but reordering
        input_gidx_to_this = torch.arange(6, 12).flip(0)
        output_gidx_on_this = torch.arange(6, 12)
    elif local_rank == 2:
        # Only recvs, also reordering
        input_gidx_to_this = vec(12, 13, 14, 15, 16, 17, 1, 4)
        output_gidx_on_this = vec(13, 13, 13, 16, 16, 16, 14, 15)

    rewrite_reducer_instance(
        reducer, input_gidx_to_this, output_gidx_on_this,
        input_elempart, output_elempart)

    if local_rank == 0:
        assert reducer.easier_reordering_selector_idx is None
        assert torch.equal(reducer.idx, vec(2, 2, 3, 3))
        assert_tensor_list_equal(reducer.runtime_halos_local_idxes, [
            vec(0, 2, 3, 5),
            vec(),
            vec(1, 4)
        ])
    elif local_rank == 1:
        assert torch.equal(
            reducer.easier_reordering_selector_idx,  # type: ignore
            torch.arange(6).flip(0))
        assert torch.equal(reducer.idx, torch.arange(6))
        assert_tensor_list_equal(reducer.runtime_halos_local_idxes, [
            vec(),
            torch.arange(6),
            vec()
        ])
    elif local_rank == 2:
        assert torch.equal(
            reducer.easier_reordering_selector_idx,  # type: ignore
            vec(2, 3, 4, 0, 1, 5, 6, 7))
        assert torch.equal(reducer.idx, vec(1, 1, 1, 2, 3, 4, 4, 4))
        assert_tensor_list_equal(reducer.runtime_halos_local_idxes, [
            vec(),
            vec(),
            torch.arange(6)
        ])

    _JitRuntimeDistEnv.init_runtime_dist_env('cpu')
    elemparts = {m.v.easier_tensor_group: output_elempart}
    HaloExchangerInserter(
        [jm], [graph], elemparts,  # type: ignore
    ).run()
    ReorderingSelectorInserter([jm], [graph]).run()

    call_haloxchg: Node
    call_reordering_selector: Node
    haloxchg: HaloExchanger
    if local_rank == 0:
        getattr_v, call_haloxchg, call_reducer, out = graph.nodes

        haloxchg = jm.get_submodule(call_haloxchg.target)  # type: ignore
        assert haloxchg.concat_buffer_length == 4

    elif local_rank == 1:
        getattr_v, call_reordering_selector, call_reducer, \
            out = graph.nodes

        assert call_reordering_selector.op == FX.CALL_MODULE
        assert isinstance(
            jm.get_submodule(call_reordering_selector.target),  # type: ignore
            easier.Selector)

    elif local_rank == 2:
        getattr_v, call_haloxchg, call_reordering_selector, call_reducer, \
            out = graph.nodes

        haloxchg = jm.get_submodule(call_haloxchg.target)  # type: ignore
        assert haloxchg.concat_buffer_length == 8

        assert isinstance(
            jm.get_submodule(call_reordering_selector.target),  # type: ignore
            easier.Selector)

    assert getattr_v.op == FX.GET_ATTR
    assert call_reducer.op == FX.CALL_MODULE
    assert isinstance(
        jm.get_submodule(call_reducer.target),  # type: ignore
        easier.Reducer)


def test_halo_exchanger_insertion_for_reducer():
    mpirun_singlenode(3, worker__test_halo_exchanger_insertion_for_reducer)
