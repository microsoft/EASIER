# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing_extensions import Literal
from unittest.mock import patch
import torch
import pytest
import contextlib

import h5py
import tempfile
import os

from easier.examples.models import Poisson
import easier as esr

from ..utils import \
    torchrun_singlenode, get_random_str, assert_tensor_list_equal, \
    when_ngpus_ge_2


class Model(esr.Module):
    def __init__(self, nf, device='cpu') -> None:
        super().__init__()
        eqn = Poisson(100, device)
        nv = self.nv = eqn.x.shape[0]
        ne = self.ne = eqn.src.shape[0]

        self.reducer_src = esr.Reducer(eqn.src, nv)
        self.reducer_dst = esr.Reducer(eqn.dst, nv)
        self.selector_src = esr.Selector(eqn.src)
        self.selector_dst = esr.Selector(eqn.dst)

        self.vertex_tensor = esr.Tensor(
            torch.randn((nv, nf)).to(device=device), mode='partition')
        self.edge_tensor = esr.Tensor(torch.randn(
            (ne, nf)).to(device=device), mode='partition')
        self.tensor = esr.Tensor(torch.randn(
            (1, nf)).to(device=device), mode='replicate')

    def forward(self):
        dst = self.selector_dst(self.vertex_tensor)
        src = self.selector_src(self.vertex_tensor)
        res = self.reducer_dst(0.5 * (dst + src) -
                               self.edge_tensor + self.tensor)
        res = self.reducer_src(0.5 * (dst + src) -
                               self.edge_tensor + self.tensor) + res

        self.vertex_tensor[:] = res
        self.edge_tensor[:] = (dst + src) * 0.5
        self.tensor[:] = esr.sum(self.vertex_tensor) / self.nv


class Model2(esr.Module):
    def __init__(self, m1: Model, nf):
        super().__init__()
        nv = m1.nv

        self.reducer = m1.reducer_dst
        self.selector_src = m1.selector_src
        self.selector_dst = esr.Selector(torch.ones_like(m1.selector_src.idx))

        self.tensor1 = m1.vertex_tensor
        self.tensor2 = esr.Tensor(
            torch.randn((nv, nf)).to(device=m1.vertex_tensor.device),
            mode='partition')

    def forward(self):
        dst = self.selector_dst(self.tensor1)
        src = self.selector_src(self.tensor2)
        res = self.reducer(0.5 * (dst + src))

        self.tensor1[:] = res
        self.tensor2[:] = res * 1.5


def _equal_jitted_selector(s1: esr.Selector, s2: esr.Selector):
    assert torch.equal(s1.idx, s2.idx)
    assert s1.easier_index_status == s2.easier_index_status == 'rewritten'
    assert s1.runtime_halos_recv_lengths == s2.runtime_halos_recv_lengths
    assert_tensor_list_equal(
        s1.runtime_halos_local_idxes, s2.runtime_halos_local_idxes
    )


def _equal_jitted_reducer(r1: esr.Reducer, r2: esr.Reducer):
    assert torch.equal(r1.idx, r2.idx)
    assert r1.easier_index_status == r2.easier_index_status == 'rewritten'
    assert r1.runtime_halos_recv_lengths == r2.runtime_halos_recv_lengths
    assert_tensor_list_equal(
        r1.runtime_halos_local_idxes, r2.runtime_halos_local_idxes
    )

    assert r1.n == r2.n


def worker__test_jitted_dump(
    local_rank: int, world_size: int, dev_type: str, dumpdir: str
):
    torch.manual_seed(2345)
    model_dev = torch.device(dev_type)

    m = Model(3, model_dev)  # type: ignore

    jm1, = esr.compile([m], 'torch', partition_mode='evenly')  # type: ignore
    esr.dump([jm1], dumpdir)
    jm1: Model

    torch.manual_seed(2345)
    m = Model(3, model_dev)  # type: ignore
    jm2, = esr.compile(
        [m], 'torch', load_dir=dumpdir, partition_mode='evenly'  # type: ignore
    )
    jm2: Model

    _equal_jitted_selector(jm1.selector_src, jm2.selector_src)
    _equal_jitted_selector(jm1.selector_dst, jm2.selector_dst)
    _equal_jitted_selector(
        getattr(jm1, 'csr_selector0reducer_src'),
        getattr(jm2, 'csr_selector0reducer_src')
    )
    _equal_jitted_reducer(jm1.reducer_src, jm2.reducer_src)
    _equal_jitted_reducer(jm1.reducer_dst, jm2.reducer_dst)

    _equal_jitted_selector(
        getattr(jm1, 'reordering_selector0reducer_dst'),
        getattr(jm2, 'reordering_selector0reducer_dst'),
    )
    _equal_jitted_selector(
        getattr(jm1, 'reordering_selector1reducer_src'),
        getattr(jm2, 'reordering_selector1reducer_src'),
    )


def worker__test_jitted_shared(
    local_rank: int, world_size: int, dev_type: str, dumpdir: str
):
    torch.manual_seed(2345)
    model_dev = torch.device(dev_type)

    m1 = Model(3, model_dev)  # type: ignore
    m2 = Model2(m1, 3)
    # Only evenly partition, cause reordering Selectors to appear.
    jm1a, jm2a = esr.compile(
        [m1, m2], 'torch', partition_mode='evenly'
    )  # type: ignore
    esr.dump([jm1a, jm2a], dumpdir)
    jm1a: Model
    jm2a: Model2

    torch.manual_seed(2345)
    m1 = Model(3, model_dev)  # type: ignore
    m2 = Model2(m1, 3)
    jm1b, jm2b = esr.compile(
        [m1, m2], 'torch', load_dir=dumpdir, partition_mode='evenly'
    )  # type: ignore
    jm1b: Model
    jm2b: Model2

    _equal_jitted_selector(jm1a.selector_src, jm1b.selector_src)
    _equal_jitted_selector(jm1a.selector_dst, jm1b.selector_dst)
    _equal_jitted_selector(
        getattr(jm1a, 'csr_selector0reducer_src'),
        getattr(jm1b, 'csr_selector0reducer_src')
    )
    _equal_jitted_reducer(jm1a.reducer_src, jm1b.reducer_src)
    _equal_jitted_reducer(jm1a.reducer_dst, jm1b.reducer_dst)

    _equal_jitted_selector(jm2a.selector_dst, jm2b.selector_dst)

    _equal_jitted_selector(
        getattr(jm1a, 'reordering_selector0reducer_dst'),
        getattr(jm1b, 'reordering_selector0reducer_dst'),
    )
    _equal_jitted_selector(
        getattr(jm1a, 'reordering_selector1reducer_src'),
        getattr(jm1b, 'reordering_selector1reducer_src'),
    )


@pytest.mark.parametrize('dev_type', [
    'cpu',
    pytest.param('cuda', marks=when_ngpus_ge_2)
])
class TestModuleDump:
    """
    Run using `pytest -s` to see logs of where the dump jit.hdf5 is stored.
    """

    def test_jitted_dump(self, dev_type):
        dumpdir = os.path.join(tempfile.gettempdir(), "easier", "tests",
                               get_random_str())
        torchrun_singlenode(2, worker__test_jitted_dump, (dev_type, dumpdir,))

    def test_jitted_shared(self, dev_type):
        dumpdir = os.path.join(tempfile.gettempdir(), "easier", "tests",
                               get_random_str())
        torchrun_singlenode(2, worker__test_jitted_shared, (dev_type, dumpdir,))
