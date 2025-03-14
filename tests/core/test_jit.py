# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from unittest.mock import patch
import torch
from torch.fx.graph_module import GraphModule
import pytest
import tempfile
import os
import h5py

import easier as esr
from easier.core.jit import EasierTracer
from easier.core.passes.utils import OrderedSet
from easier.examples import Poisson
from easier.core.runtime.dist_env import DummyDistEnv

from ..utils import torchrun_singlenode, get_random_str


class Model(esr.Module):
    def __init__(self, nf, device='cpu') -> None:
        super().__init__()
        eqn = Poisson(100, device)
        nv = self.nv = eqn.x.shape[0]
        ne = self.ne = eqn.src.shape[0]

        self.reducer = esr.Reducer(eqn.dst, nv)
        self.selector_src = esr.Selector(eqn.src)
        self.selector_dst = esr.Selector(eqn.dst)

        self.vertex_tensor = esr.Tensor(
            torch.randn((nv, nf)).to(device=device), mode='partition')
        self.edge_tensor = esr.Tensor(torch.randn(
            (ne, nf)).to(device=device), mode='partition')
        self.tensor = esr.Tensor(torch.randn(
            (1, nf)).to(device=device), mode='replicate')

        self.out1 = esr.Tensor(torch.zeros(nf).to(
            device=device), mode='replicate')

    def forward(self):
        dst = self.selector_dst(self.vertex_tensor)
        src = self.selector_src(self.vertex_tensor)
        res = self.reducer(0.5 * (dst + src) - self.edge_tensor + self.tensor)

        self.vertex_tensor[:] = res
        self.edge_tensor[:] = (dst + src) * 0.5
        self.tensor[:] = esr.sum(self.vertex_tensor) / self.nv
        self.out1[:] = esr.norm(res, 2)[0]


@pytest.mark.usefixtures('dummy_dist_env')
def test_jit_nnModule():
    """test whether easier ops are jitted as leaf node"""
    m = Model(3)

    tracer = EasierTracer()
    graph = tracer.trace(m)
    jitted = GraphModule(m, graph)

    for node in jitted.graph.nodes:
        if node.target == 'selector_src':
            assert isinstance(getattr(jitted, node.target), esr.Selector)
        elif node.target == 'selector_dst':
            assert isinstance(getattr(jitted, node.target), esr.Selector)
        elif node.target == 'reducer':
            assert isinstance(
                getattr(jitted, node.target), esr.Reducer)


@pytest.mark.usefixtures('dummy_dist_env')
def test_jit_tensors():
    """test whether easier tensors are mutable after jitted"""

    torch.manual_seed(2345)
    nonjitted, = esr.compile([Model(3)], backend='none')  # type: ignore
    for _ in range(5):
        nonjitted()
    nonjitted_res1 = nonjitted.out1.clone()

    torch.manual_seed(2345)
    jitted, = esr.compile([Model(3)], backend='none')  # type: ignore
    jitted: Model
    for _ in range(5):
        jitted()
    jitted_res1 = jitted.out1.clone()

    assert torch.all(torch.isclose(
        jitted_res1, nonjitted_res1))  # type: ignore


@pytest.mark.usefixtures('dummy_dist_env')
def test_jit_orphan_tensors():
    """
    Test that Tensors that are not involved in Selector-Reducer pairs
    are properly partitioned.
    """
    n = 10

    class M(esr.Module):
        def __init__(self):
            super().__init__()
            self.v = esr.Tensor(torch.ones(10), mode='partition')
            self.s = esr.Reducer(torch.tensor([1, 2, 3]), n)
            self.v2 = esr.Tensor(torch.ones(13), mode='partition')

        def forward(self):
            self.v[:] = self.v + 3

    m = M()
    jitted, = esr.compile([m])  # type: ignore
    jitted: M
    jitted()
    v = jitted.v.collect()
    assert torch.equal(torch.full_like(v, 4), v)

    assert jitted.v.easier_tensor_group is not None
    assert jitted.v.easier_tensor_group.tensor_defs == OrderedSet([jitted.v])
    assert jitted.v.elempart.lengths[0] == 10  # type: ignore

    v2 = jitted.v2.collect()
    assert torch.equal(torch.full_like(v2, 1), v2)

    assert jitted.v2.easier_tensor_group is not None
    assert jitted.v2.easier_tensor_group.tensor_defs == OrderedSet([jitted.v2])
    assert jitted.v2.elempart.lengths[0] == 13  # type: ignore


@pytest.mark.usefixtures('dummy_dist_env')
def test_nested_easier_modules():
    class Inner(esr.Module):
        def __init__(self):
            super().__init__()
            self.v = esr.Tensor(torch.ones(10), mode='partition')
            self.s = esr.Selector(torch.arange(10))
            self.r = esr.Reducer(torch.arange(10), 10)

        def forward(self):
            self.v[:] = self.r(self.s(self.v))

    class Outer(esr.Module):
        def __init__(self):
            super().__init__()
            self.inner = Inner()
            self.s = esr.Selector(torch.arange(10))
            self.r = esr.Reducer(torch.arange(10), 10)

        def forward(self):
            k = self.s(self.inner.v)
            self.inner()
            self.inner.v[:] = self.r(k)

    outer = Outer()
    j_outer, = esr.compile([outer])

    g: Graph = j_outer.forward.__self__.graph  # type: ignore
    for n in g.nodes:
        if n.op == 'call_module' and n.target == 'inner':
            break
    else:
        assert False, "shoud have nested call"


def worker__test_collect(local_rank: int, world_size: int,
                         dev_type: str, jit_backend: str):
    if dev_type == 'cpu':
        model_dev = 'cpu'
    else:
        model_dev = f'{dev_type}:{local_rank}'
    model_dev = torch.device(model_dev)

    torch.manual_seed(2345)
    m, = esr.compile(
        [Model(3, model_dev)], backend='none')
    m()

    orig_vertex = m.vertex_tensor.clone().cpu()
    orig_edge = m.edge_tensor.clone().cpu()
    orig_replica = m.tensor.clone().cpu()

    torch.manual_seed(2345)
    jitted, = esr.compile(
        [Model(3, model_dev)], backend=jit_backend)  # type: ignore
    jitted: Model
    jitted()

    # Simple test that partition is really done.
    assert jitted.vertex_tensor.shape[0] < orig_vertex.shape[0]

    collected_vertex = jitted.vertex_tensor.collect()
    collected_edge = jitted.edge_tensor.collect()
    collected_replica = jitted.tensor.collect()
    assert collected_vertex.device == model_dev
    assert collected_edge.device == model_dev
    assert collected_vertex.device == model_dev
    torch.testing.assert_close(collected_vertex.cpu(), orig_vertex)
    torch.testing.assert_close(collected_edge.cpu(), orig_edge)
    torch.testing.assert_close(collected_replica.cpu(), orig_replica)

    if jit_backend == 'torch':
        comm_dev_type = model_dev.type
    elif jit_backend == 'cpu':
        comm_dev_type = 'cpu'
    elif jit_backend == 'gpu':
        comm_dev_type = 'cuda'

    from easier.core.runtime.dist_env import get_runtime_dist_env
    if comm_dev_type == 'cpu':
        assert get_runtime_dist_env().comm_device.type == 'cpu'
    elif comm_dev_type == 'cuda':
        assert get_runtime_dist_env().comm_device.type == 'cuda'
        assert get_runtime_dist_env().comm_device.index == local_rank
    else:
        assert False


def worker__test_save(local_rank: int, world_size: int,
                      dev_type: str):
    model_dev = torch.device(dev_type)

    torch.manual_seed(2345)
    m, = esr.compile([Model(3, model_dev)], backend='none')
    m()

    orig_vertex = m.vertex_tensor.data.to(device='cpu', copy=True)
    orig_edge = m.edge_tensor.data.to(device='cpu', copy=True)
    orig_replica = m.tensor.data.to(device='cpu', copy=True)

    torch.manual_seed(2345)
    jitted, = esr.compile(
        [Model(3, model_dev)], backend='torch')  # type: ignore
    jitted: Model
    jitted()

    # Simple test that partition is really done.
    assert jitted.vertex_tensor.shape[0] < orig_vertex.shape[0]

    if local_rank == 0:
        fn = get_random_str() + ".hdf5"
        dir = os.path.join(tempfile.gettempdir(), "easier", "tests")
        os.makedirs(dir, exist_ok=True)
        fpath = os.path.join(dir, fn)
    else:
        fpath = None

    jitted.vertex_tensor.save(fpath, 'vertex')
    jitted.edge_tensor.save(fpath, 'edge')
    jitted.tensor.save(fpath, 'replica')

    if local_rank == 0:
        with h5py.File(fpath, 'r') as h5f:
            torch.testing.assert_close(
                torch.from_numpy(h5f['vertex'][:]), orig_vertex)
            torch.testing.assert_close(
                torch.from_numpy(h5f['edge'][:]), orig_edge)
            torch.testing.assert_close(
                torch.from_numpy(h5f['replica'][:]), orig_replica)


when_ngpus_ge_2 = pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="no enough CUDA GPU (ngpus >= 2) to test distribution")


class TestJittedUsage:

    @pytest.mark.parametrize('dev_type', [
        'cpu',
        pytest.param('cuda', marks=when_ngpus_ge_2)
    ])
    @pytest.mark.parametrize('jit_backend', [
        'torch',
        'cpu',
        pytest.param('gpu', marks=when_ngpus_ge_2)
    ])
    def test_collect(self, dev_type: str, jit_backend: str):
        if jit_backend == 'torch':
            init_type = dev_type
        elif jit_backend == 'cpu':
            init_type = 'cpu'
        elif jit_backend == 'gpu':
            init_type = 'cuda'
        else:
            assert False
        torchrun_singlenode(
            2, worker__test_collect,
            (dev_type, jit_backend),
            init_type=init_type  # type: ignore
        )

    @pytest.mark.parametrize('dev_type', [
        'cpu',
        pytest.param('cuda', marks=when_ngpus_ge_2)
    ])
    def test_save(self, dev_type: str):
        torchrun_singlenode(
            2, worker__test_save, (dev_type,),
            init_type=dev_type  # type: ignore
        )
