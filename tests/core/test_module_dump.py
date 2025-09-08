# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import contextlib
from unittest.mock import patch
import torch
import pytest

import tempfile
import os

import easier as esr
from easier.core.dump import load_dumps as _orig_load_dumps
from easier.core.runtime.data_loader.factories import InMemoryTensorLoader
from easier.core.runtime.data_loader.ops import StridedDataLoader
from easier.core.utils import get_random_str

from ..utils import \
    torchrun_singlenode, assert_tensor_list_equal, \
    when_ngpus_ge_2, mpi_e2e, mpirun_singlenode, \
    import_poisson, MESH, POISSON
from tests.core.utils import multi_stage_zero_length_partition

Poisson = import_poisson()


class Model(esr.Module):
    def __init__(self, nf, device='cpu') -> None:
        super().__init__()
        eqn = Poisson(MESH, POISSON, device)
        nv = self.nv = eqn.x.shape[0]
        ne = self.ne = eqn.src.shape[0]

        self.reducer_src = esr.Reducer(eqn.src, nv)
        self.reducer_dst = esr.Reducer(eqn.dst, nv)
        self.selector_src = esr.Selector(eqn.src)
        self.selector_dst = esr.Selector(eqn.dst)

        dl = InMemoryTensorLoader(
            torch.randn((nv, nf)).to(device=device)
        )[:, :]
        assert isinstance(dl, StridedDataLoader)

        self.vertex_tensor = esr.Tensor(dl, mode='partition')
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


@contextlib.contextmanager
def expect_load_succeeds():
    def _stub_load_dumps(*args, **kwargs):
        loaded_graphs = _orig_load_dumps(*args, **kwargs)
        assert loaded_graphs != None
        return loaded_graphs
    with patch(
        f'{esr.compile.__module__}.{_orig_load_dumps.__name__}',
        new=_stub_load_dumps
    ) as mock_load_dumps:
        yield


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
    with expect_load_succeeds():
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
    with expect_load_succeeds():
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


@pytest.mark.parametrize('xrun_singlenode', [
    torchrun_singlenode,
    pytest.param(mpirun_singlenode, marks=mpi_e2e)
])
@pytest.mark.parametrize('dev_type', [
    'cpu',
    pytest.param('cuda', marks=when_ngpus_ge_2)
])
class TestModuleDump:
    """
    Run using `pytest -s` to see logs of where the dump jit.hdf5 is stored.
    """

    def test_jitted_dump(self, xrun_singlenode, dev_type):
        dumpdir = os.path.join(tempfile.gettempdir(), "easier", "tests",
                               get_random_str())
        xrun_singlenode(
            2, worker__test_jitted_dump, (dev_type, dumpdir,),
            init_type=dev_type
        )

    def test_jitted_shared(self, xrun_singlenode, dev_type):
        dumpdir = os.path.join(tempfile.gettempdir(), "easier", "tests",
                               get_random_str())
        xrun_singlenode(
            2, worker__test_jitted_shared,
            (dev_type, dumpdir,),
            init_type=dev_type
        )


def worker__test_smoke_zerolength_dump(
    local_rank: int, world_size: int, dev_type: str, dumpdir: str
):
    model_dev = torch.device(dev_type)

    torch.manual_seed(2345)
    m = Model(3, model_dev)  # type: ignore

    with multi_stage_zero_length_partition((m.vertex_tensor, m.edge_tensor)):
        jm1, = esr.compile([m], 'torch')  # type: ignore
    esr.dump([jm1], dumpdir)
    jm1: Model
    jm1()

    orig_vertex = jm1.vertex_tensor.collect()
    orig_edge = jm1.edge_tensor.collect()
    orig_replica = jm1.tensor.collect()

    torch.manual_seed(2345)
    m = Model(3, model_dev)  # type: ignore
    with expect_load_succeeds():
        jm2, = esr.compile([m], 'torch', load_dir=dumpdir)  # type: ignore
    jm2: Model
    jm2()

    collected_vertex = jm2.vertex_tensor.collect()
    collected_edge = jm2.edge_tensor.collect()
    collected_replica = jm2.tensor.collect()
    torch.testing.assert_close(collected_vertex, orig_vertex)
    torch.testing.assert_close(collected_edge, orig_edge)
    torch.testing.assert_close(collected_replica, orig_replica)


@pytest.mark.parametrize('dev_type', [
    'cpu',
    pytest.param('cuda', marks=when_ngpus_ge_2)
])
def test_smoke_zerolength_dump(dev_type):
    dumpdir = os.path.join(
        tempfile.gettempdir(), "easier", "tests", get_random_str()
    )
    torchrun_singlenode(
        4 if dev_type == 'cpu' else 2,
        worker__test_smoke_zerolength_dump,
        (dev_type, dumpdir),
        init_type=dev_type
    )
