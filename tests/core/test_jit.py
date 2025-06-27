# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Union
from unittest.mock import patch
import torch
from torch.fx.graph_module import GraphModule
import pytest
import tempfile
import os
import h5py

import easier as esr
from easier.core.jit import EasierTracer
from easier.core.utils import get_random_str
from easier.core.passes.utils import OrderedSet

from tests.utils import \
    torchrun_singlenode, mpi_e2e, mpirun_singlenode, when_ngpus_ge_2, \
    import_poisson, MESH, POISSON
from tests.core.utils import multi_stage_zero_length_partition

Poisson = import_poisson()


class Model(esr.Module):
    def __init__(self, nf, device: Union[str, torch.device] = 'cpu') -> None:
        super().__init__()
        eqn = Poisson(MESH, POISSON, device)  # type: ignore
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
    jitted, = esr.compile([m], 'torch')  # type: ignore
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
    j_outer, = esr.compile([outer], 'torch')

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
    m()

    from easier.core.runtime.dist_env import get_default_dist_env
    def_dist_env = get_default_dist_env()
    if def_dist_env.rank == 0:
        orig_vertex = m.vertex_tensor.clone().cpu()
        orig_edge = m.edge_tensor.clone().cpu()
        orig_replica = m.tensor.clone().cpu()
        def_dist_env.broadcast_object_list(
            0, [orig_vertex, orig_edge, orig_replica]
        )
    else:
        [orig_vertex, orig_edge, orig_replica] = \
            def_dist_env.broadcast_object_list(0)

    torch.manual_seed(2345)
    jitted, = esr.compile(
        [Model(3, model_dev)], backend=jit_backend)  # type: ignore
    jitted: Model
    jitted()
    jitted()

    # Simple test that partition is really done.
    assert jitted.vertex_tensor.shape[0] < orig_vertex.shape[0]

    collected_vertex = jitted.vertex_tensor.collect()
    collected_edge = jitted.edge_tensor.collect()
    collected_replica = jitted.tensor.collect()
    assert collected_vertex.device == model_dev
    assert collected_edge.device == model_dev
    assert collected_replica.device == model_dev
    torch.testing.assert_close(collected_vertex.cpu(), orig_vertex)
    torch.testing.assert_close(collected_edge.cpu(), orig_edge)
    torch.testing.assert_close(collected_replica.cpu(), orig_replica)

    if jit_backend == 'torch':
        comm_dev_type = model_dev.type
    else:
        comm_dev_type = jit_backend

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
    m()

    orig_vertex = m.vertex_tensor.data.to(device='cpu', copy=True)
    orig_edge = m.edge_tensor.data.to(device='cpu', copy=True)
    orig_replica = m.tensor.data.to(device='cpu', copy=True)

    torch.manual_seed(2345)
    jitted, = esr.compile(
        [Model(3, model_dev)], backend='torch'  # type: ignore
    )
    jitted: Model
    jitted()
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

    from easier.core.module import _dist_save as _orig_dist_save

    def _dist_save_with_chunk_size(tensor, h5d, *, chunk_size=None):
        # Test when chuck size is smaller than the dataset size.
        return _orig_dist_save(tensor, h5d, chunk_size=13)

    with patch(f'{_orig_dist_save.__module__}._dist_save') as mock_dist_save:
        mock_dist_save.side_effect = _dist_save_with_chunk_size

        jitted.vertex_tensor.save(fpath, 'vertex')
        jitted.edge_tensor.save(fpath, 'edge')
        jitted.tensor.save(fpath, 'replica')

        # replica.save() does not call _dist_save()
        assert mock_dist_save.call_count == 2

    if local_rank == 0:
        with h5py.File(fpath, 'r') as h5f:
            torch.testing.assert_close(
                torch.from_numpy(h5f['vertex'][:]), orig_vertex)
            torch.testing.assert_close(
                torch.from_numpy(h5f['edge'][:]), orig_edge)
            torch.testing.assert_close(
                torch.from_numpy(h5f['replica'][:]), orig_replica)


@pytest.mark.parametrize('xrun_singlenode', [
    torchrun_singlenode,
    pytest.param(mpirun_singlenode, marks=mpi_e2e)
])
class TestJittedUsage:

    @pytest.mark.parametrize('dev_type', [
        'cpu',
        pytest.param('cuda', marks=when_ngpus_ge_2)
    ])
    @pytest.mark.parametrize('jit_backend', [
        'torch',
        'cpu',
        pytest.param('cuda', marks=when_ngpus_ge_2)
    ])
    def test_collect(
        self, xrun_singlenode, dev_type: str, jit_backend: str
    ):
        if jit_backend == 'torch':
            init_type = dev_type
        else:
            init_type = jit_backend
        xrun_singlenode(
            2, worker__test_collect,
            (dev_type, jit_backend),
            init_type=init_type  # type: ignore
        )

    @pytest.mark.parametrize('dev_type', [
        'cpu',
        pytest.param('cuda', marks=when_ngpus_ge_2)
    ])
    def test_save(self, xrun_singlenode, dev_type: str):
        xrun_singlenode(
            2, worker__test_save, (dev_type,),
            init_type=dev_type  # type: ignore
        )


def worker__test_none_collect(local_rank: int, world_size: int, dev_type: str):
    torch.manual_seed(2345)
    if dev_type == 'cpu':
        model_dev = 'cpu'
    else:
        model_dev = f'{dev_type}:{local_rank}'
    model_dev = torch.device(model_dev)
    m = Model(3, model_dev)

    from h5py import File as _orig_h5py_File
    _orig_m_forward = m.forward

    with patch(f'h5py.File') as mock_h5py_File, \
            patch.object(m, 'forward') as mock_m_forward:

        mock_h5py_File.side_effect = _orig_h5py_File
        mock_m_forward.side_effect = _orig_m_forward

        [jitted] = esr.compile([m], backend='none')  # type: ignore
        jitted: Model
        jitted()
        jitted()

    if local_rank == 0:
        assert mock_h5py_File.call_count > 0
        assert mock_m_forward.call_count == 2
    else:
        assert mock_h5py_File.call_count == 0
        assert mock_m_forward.call_count == 0

    # backend='none' also moves data to proper devices
    def _assert_device(tensor: torch.Tensor):
        if dev_type == 'cpu':
            assert tensor.device.type == 'cpu'
        else:
            jitted_dev = torch.device(dev_type, local_rank)
            assert tensor.device == jitted_dev
    _assert_device(jitted.vertex_tensor)
    _assert_device(jitted.edge_tensor)
    _assert_device(jitted.tensor)

    # collect() moves to original device on each rank, but because
    # esr.init() would call torch.cuda.set_device(local_rank),
    # collected tensors are on individual devices.
    collected_vertex = jitted.vertex_tensor.collect()
    collected_edge = jitted.edge_tensor.collect()
    collected_replica = jitted.tensor.collect()
    _assert_device(collected_vertex)
    _assert_device(collected_edge)
    _assert_device(collected_replica)

    from easier.core.runtime.utils import check_collective_equality

    def _assert_coll_eq(tensor: torch.Tensor):
        check_collective_equality('collected', tensor.cpu(), torch.equal)

    _assert_coll_eq(collected_vertex)
    _assert_coll_eq(collected_edge)
    _assert_coll_eq(collected_replica)


def worker__test_none_save(local_rank: int, world_size: int, dev_type: str):
    model_dev = torch.device(dev_type)

    torch.manual_seed(2345)
    m = Model(3, model_dev)

    from h5py import File as _orig_h5py_File
    _orig_m_forward = m.forward

    with patch(f'h5py.File') as mock_h5py_File, \
            patch.object(m, 'forward') as mock_m_forward:

        mock_h5py_File.side_effect = _orig_h5py_File
        mock_m_forward.side_effect = _orig_m_forward

        [jitted] = esr.compile([m], backend='none')  # type: ignore
        jitted: Model
        jitted()
        jitted()

    if local_rank == 0:
        assert mock_h5py_File.call_count > 0
        assert mock_m_forward.call_count == 2
    else:
        assert mock_h5py_File.call_count == 0
        assert mock_m_forward.call_count == 0

    collected_vertex = jitted.vertex_tensor.collect().cpu()
    collected_edge = jitted.edge_tensor.collect().cpu()
    collected_replica = jitted.tensor.collect().cpu()

    if local_rank == 0:
        fn = get_random_str() + ".hdf5"
        dir = os.path.join(tempfile.gettempdir(), "easier", "tests")
        os.makedirs(dir, exist_ok=True)
        fpath = os.path.join(dir, fn)
    else:
        fpath = None

    from easier.core.module import _dist_save as _orig_dist_save
    from h5py import File as _orig_h5py_File

    def _dist_save_with_chunk_size(tensor, h5d, *, chunk_size=None):
        # Test when chuck size is smaller than the dataset size.
        return _orig_dist_save(tensor, h5d, chunk_size=13)

    with patch(f'{_orig_dist_save.__module__}._dist_save') as mock_dist_save, \
            patch(f'h5py.File') as mock_h5py_File:

        mock_dist_save.side_effect = _dist_save_with_chunk_size
        mock_h5py_File.side_effect = _orig_h5py_File

        jitted.vertex_tensor.save(fpath, 'vertex')
        jitted.edge_tensor.save(fpath, 'edge')
        jitted.tensor.save(fpath, 'replica')

        # in backend=='none', mode=='partitioned' tensor won't call dist_save()
        assert mock_dist_save.call_count == 0

        # ranks other than rank-0 won't do writing
        if local_rank == 0:
            assert mock_h5py_File.call_count == 3

        else:
            assert mock_h5py_File.call_count == 0

    if local_rank == 0:
        with h5py.File(fpath, 'r') as h5f:
            torch.testing.assert_close(
                torch.from_numpy(h5f['vertex'][:]), collected_vertex
            )
            torch.testing.assert_close(
                torch.from_numpy(h5f['edge'][:]), collected_edge
            )
            torch.testing.assert_close(
                torch.from_numpy(h5f['replica'][:]), collected_replica
            )


@pytest.mark.parametrize('xrun_singlenode', [
    torchrun_singlenode,
    pytest.param(mpirun_singlenode, marks=mpi_e2e)
])
@pytest.mark.parametrize('dev_type', [
    'cpu',
    pytest.param('cuda', marks=when_ngpus_ge_2)
])
class TestBackendNoneUsage:
    def test_collect(self, xrun_singlenode, dev_type: str):
        xrun_singlenode(
            2, worker__test_none_collect, (dev_type,), init_type=dev_type
        )

    def test_save(self, xrun_singlenode, dev_type: str):
        xrun_singlenode(
            2, worker__test_none_save, (dev_type,), init_type=dev_type
        )


def worker__test_zerolength_collect(local_rank: int, world_size: int, dev_type):
    torch.manual_seed(2345)
    m = Model(3, 'cpu')
    [m] = esr.compile([m], backend='none')
    m()
    m()

    from easier.core.runtime.dist_env import get_default_dist_env
    def_dist_env = get_default_dist_env()
    if def_dist_env.rank == 0:
        orig_vertex = m.vertex_tensor.clone().cpu()
        orig_edge = m.edge_tensor.clone().cpu()
        orig_replica = m.tensor.clone().cpu()
        def_dist_env.broadcast_object_list(
            0, [orig_vertex, orig_edge, orig_replica]
        )
    else:
        [orig_vertex, orig_edge, orig_replica] = \
            def_dist_env.broadcast_object_list(0)

    torch.manual_seed(2345)
    m = Model(3, 'cpu')

    with multi_stage_zero_length_partition((m.vertex_tensor, m.edge_tensor)):
        [jitted] = esr.compile([m], backend=dev_type)  # type: ignore
    jitted: Model
    jitted()
    jitted()

    # Simple test that partition is really done.
    if local_rank == 0:
        assert jitted.vertex_tensor.shape[0] == 0
        assert jitted.edge_tensor.shape[0] == 0

    collected_vertex = jitted.vertex_tensor.collect()
    collected_edge = jitted.edge_tensor.collect()
    collected_replica = jitted.tensor.collect()
    torch.testing.assert_close(collected_vertex, orig_vertex)
    torch.testing.assert_close(collected_edge, orig_edge)
    torch.testing.assert_close(collected_replica, orig_replica)


def worker__test_zerolength_save(local_rank: int, world_size: int, dev_type):
    torch.manual_seed(2345)
    m, = esr.compile([Model(3, 'cpu')], backend='none')
    m()
    m()

    orig_vertex = m.vertex_tensor.data.to(device='cpu', copy=True)
    orig_edge = m.edge_tensor.data.to(device='cpu', copy=True)
    orig_replica = m.tensor.data.to(device='cpu', copy=True)

    torch.manual_seed(2345)
    m = Model(3, 'cpu')

    with multi_stage_zero_length_partition((m.vertex_tensor, m.edge_tensor)):
        [jitted] = esr.compile([m], backend=dev_type)  # type: ignore
    jitted: Model
    jitted()
    jitted()

    # Simple test that partition is really done.
    if local_rank == 0:
        assert jitted.vertex_tensor.shape[0] == 0
        assert jitted.edge_tensor.shape[0] == 0

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


class NotFullModel(esr.Module):
    def __init__(self):
        super().__init__()

        self.vertex = esr.Tensor(
            torch.arange(2, 38).reshape(-1, 2).double(), mode='partition'
        )
        self.edge = esr.Tensor(
            torch.arange(2, 12).reshape(-1, 2).double(), mode='partition'
        )
        self.selector = esr.Selector(torch.arange(5) // 2)
        self.reducer = esr.Reducer(torch.ones(5, dtype=torch.int64), n=18)
        self.replica = esr.Tensor(
            torch.zeros([1, 2]).double(), mode='replicate'
        )

    def forward(self):
        self.edge[:] += self.selector(self.vertex)
        self.edge[:] += torch.einsum('ij,ij->ij', self.edge, self.edge)

        self.vertex[:] += self.reducer(self.edge)
        self.vertex[:] += torch.einsum('ij,ij->ij', self.vertex, self.vertex)

        self.replica[:] \
            = esr.sum(self.edge) * 1.2 \
            + esr.prod(self.edge) * 2.3 \
            + esr.max(self.edge) * 3.4 \
            + esr.min(self.edge) * 4.5 \
            + esr.norm(self.edge, p=2)


def worker__test_smoke_zerolength_notfull(local_rank, world_size, dev_type):
    m = NotFullModel()
    [jitted] = esr.compile([m], backend='none')  # type: ignore
    jitted: NotFullModel
    jitted()
    jitted()

    from easier.core.runtime.dist_env import get_default_dist_env
    def_dist_env = get_default_dist_env()
    if def_dist_env.rank == 0:
        orig_vertex = jitted.vertex.clone().cpu()
        orig_edge = jitted.edge.clone().cpu()
        orig_replica = jitted.replica.clone().cpu()
        def_dist_env.broadcast_object_list(
            0, [orig_vertex, orig_edge, orig_replica]
        )
    else:
        [orig_vertex, orig_edge, orig_replica] = \
            def_dist_env.broadcast_object_list(0)

    m = NotFullModel()
    [jitted] = esr.compile([m], backend=dev_type)  # type: ignore
    jitted()
    jitted()
    collected_v = jitted.vertex.collect().cpu()
    collected_e = jitted.edge.collect().cpu()
    collected_r = jitted.replica.collect().cpu()

    torch.testing.assert_close(collected_v, orig_vertex)
    torch.testing.assert_close(collected_e, orig_edge)
    torch.testing.assert_close(collected_r, orig_replica)


@pytest.mark.parametrize('dev_type', [
    'cpu',
    pytest.param('cuda', marks=when_ngpus_ge_2)
])
class TestZeroLengthPartition:
    def test_zerolength_collect(self, dev_type):
        torchrun_singlenode(
            4 if dev_type == 'cpu' else 2,
            worker__test_zerolength_collect, (dev_type,), init_type=dev_type
        )

    def test_zerolength_save(self, dev_type):
        torchrun_singlenode(
            4 if dev_type == 'cpu' else 2,
            worker__test_zerolength_save, (dev_type,), init_type=dev_type
        )

    def test_smoke_zerolength_notfull(self, dev_type):
        torchrun_singlenode(
            4 if dev_type == 'cpu' else 2,
            worker__test_smoke_zerolength_notfull,
            (dev_type,),
            init_type=dev_type
        )
