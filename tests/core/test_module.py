# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import tempfile
from typing import Union
import h5py
import pytest
import torch
import easier
from easier.core.utils import get_random_str
from tests.utils import when_ngpus_ge_2, torchrun_singlenode, have_cuda


def fully_load_data(t: easier.Tensor):
    assert not t.easier_data_ready
    is_replica = t.is_replica
    t.data = t.easier_data_loader.fully_load(
        torch.device('cpu'), replicated=is_replica
    )
    t.easier_data_ready = is_replica or 'rank0_only'


def fully_load_idx(m: Union[easier.Selector, easier.Reducer]):
    assert m.easier_index_status == 'placeholder'
    m.idx = m.easier_data_loader.fully_load(
        torch.device('cpu'), replicated=False
    )
    m.easier_index_status = 'rewritten'


@pytest.mark.usefixtures('dummy_dist_env')
class TestSelector:
    idx = torch.randint(0, 10, (20,))

    def test_selector(self):
        selector = easier.Selector(self.idx)
        fully_load_idx(selector)

        t = torch.randn((10, 1, 2))
        edge = selector(t)

        assert torch.equal(edge, t[self.idx])


@pytest.mark.usefixtures('dummy_dist_env')
class TestReducer:
    n = 3
    idx = torch.tensor([1, 1, 2, 2, 2], dtype=torch.int64)
    edge = torch.randn((5, 3, 4))

    def test_reducer(self):
        reducer = easier.Reducer(self.idx, self.n, reduce='sum')
        fully_load_idx(reducer)
        res0 = reducer(self.edge)

        res1 = torch.zeros((3, 3, 4))
        res1[1] = self.edge[0] + self.edge[1]
        res1[2] = self.edge[2] + self.edge[3] + self.edge[4]

        assert torch.equal(res0, res1)

    def test_reducer_one_dim(self):
        reducer = easier.Reducer(self.idx, self.n, reduce='sum')
        fully_load_idx(reducer)
        edge = self.edge[:, 0, 0]
        res0 = reducer(edge)

        res1 = torch.zeros(res0.shape)
        res1[1] = edge[0] + edge[1]
        res1[2] = edge[2] + edge[3] + edge[4]

        assert torch.equal(res0, res1)

    def test_reducer_out(self):
        reducer = easier.Reducer(self.idx, self.n, reduce='sum')
        fully_load_idx(reducer)
        out = torch.rand((3, 3, 4))
        out_copy = out.clone()
        res0 = reducer(self.edge, out=out)
        assert torch.equal(res0, out)

        res1 = torch.empty((3, 3, 4))
        # Reducer implies `scatter_reduce` with `include_self=False`
        # so `res1[0]` is not zeros but values left from `out`.
        res1[0] = out_copy[0]
        res1[1] = self.edge[0] + self.edge[1]
        res1[2] = self.edge[2] + self.edge[3] + self.edge[4]

        assert torch.equal(res0, res1)


@pytest.mark.usefixtures('dummy_dist_env')
@pytest.mark.parametrize('device_type', [
    'cpu',
    # no device IDs, all workers use cuda:0.
    pytest.param('cuda', marks=have_cuda)
])
def test_to(device_type: str):
    class M(easier.Module):
        def __init__(self, opt_inner: Union[None, easier.Module]) -> None:
            super().__init__()
            self.fv = easier.Tensor(
                torch.rand(3, 3, dtype=torch.float64), mode='partition'
            )
            self.iv = easier.Tensor(
                torch.arange(11, dtype=torch.int32), mode='partition'
            )

            self.replica = easier.Tensor(
                torch.rand(3, 3, dtype=torch.float64), mode='replicate'
            )
            self._constant = torch.rand(3, 3, dtype=torch.float64)

            self.reducer = easier.Reducer(torch.arange(2), 22)

            self.idt = easier.arange(13, dtype=torch.int64)
            self.fdt = easier.arange(13, dtype=torch.float32)

            self.opt_inner = opt_inner

    notcasted = M(None)

    inner = M(None)
    outer = M(inner)

    outer.to(torch.float16).to(device_type)

    def _assert(m: M):
        assert m.fv.dtype == torch.float16
        assert m.fv.device.type == device_type
        assert m.iv.dtype == torch.int32
        assert m.iv.device.type == device_type

        assert m.replica.dtype == torch.float16
        assert m.replica.device.type == device_type

        # Users are not supposed to use plain constant tensors, but use
        # replicated esr.Tensors instead.
        # FX tracing may cause plain constants to appear,
        # but not affected by .to(), dist_pass will move it.
        assert m._constant.dtype == torch.float64
        assert m._constant.device.type == 'cpu'

        assert m.reducer.idx.dtype == torch.int64
        assert m.reducer.idx.device.type == device_type

        assert m.idt.dtype == torch.int64
        assert m.idt.device.type == device_type  # int DT device is covered

        assert m.fdt.dtype == torch.float16
        assert m.fdt.device.type == device_type

    _assert(outer)
    _assert(inner)

    assert notcasted.fv.dtype == torch.float64
    assert notcasted.iv.dtype == torch.int32


def worker__test_collect_save(local_rank: int, world_size: int, dev_type: str):
    n = 3
    dev = torch.device(dev_type)

    class M(easier.Module):
        def __init__(self) -> None:
            super().__init__()
            self.v = easier.Tensor(
                torch.rand(3, 3, dtype=torch.float64, device=dev),
                mode='partition'
            )
            self.e = easier.Tensor(
                torch.rand(4, 3, dtype=torch.float64, device=dev),
                mode='partition'
            )
            self.r = easier.Tensor(
                torch.rand(9, 3, dtype=torch.float64, device=dev),
                mode='replicate'
            )

    m = M()
    [m] = easier.compile([m], backend='none')

    assert torch.equal(m.v.collect(), m.v)
    assert torch.equal(m.e.collect(), m.e)
    assert torch.equal(m.r.collect(), m.r)

    fn = get_random_str() + ".hdf5"
    dir = os.path.join(tempfile.gettempdir(), "easier", "tests")
    fpath = os.path.join(dir, fn)

    m.v.save(fpath, 'v')
    m.e.save(fpath, 'e')
    m.r.save(fpath, 'r')

    with h5py.File(fpath, 'r') as h5f:
        torch.testing.assert_close(torch.from_numpy(h5f['v'][:]), m.v.cpu())
        torch.testing.assert_close(torch.from_numpy(h5f['e'][:]), m.e.cpu())
        torch.testing.assert_close(torch.from_numpy(h5f['r'][:]), m.r.cpu())


class TestOutOfJitUsage:

    @pytest.mark.usefixtures('dummy_dist_env')
    def test_init__dtype(self):
        n = 3

        class M(easier.Module):
            def __init__(self) -> None:
                super().__init__()
                self.v = easier.Tensor(
                    torch.rand(3, 3, dtype=torch.float64), mode='partition')
                self.reducer = easier.Reducer(
                    torch.tensor([1, 2, 2, 0], dtype=torch.int64), n)
                self.e = easier.Tensor(
                    torch.rand(4, 3, dtype=torch.float64), mode='partition')
                self.r = easier.Tensor(
                    torch.rand(1, 3), mode='replicate')

        m = M()
        [m] = easier.compile([m], backend='none')

        assert m.v.dtype == torch.float64
        assert m.e.dtype == torch.float64
        assert m.r.dtype == torch.float32

        assert m.reducer.idx.dtype == torch.int64

    @pytest.mark.usefixtures('dummy_dist_env')
    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA unavailable")
    def test_init__device(self):
        n = 3

        class M(easier.Module):
            def __init__(self) -> None:
                super().__init__()
                self.v = easier.Tensor(
                    torch.rand(3, 3, dtype=torch.float64).cuda(),
                    mode='partition')
                self.reducer = easier.Reducer(
                    torch.tensor([1, 2, 2, 0], dtype=torch.int64).cuda(), n)
                self.e = easier.Tensor(
                    torch.rand(4, 3, dtype=torch.float64).cuda(),
                    mode='partition')
                self.r = easier.Tensor(
                    torch.rand(1, 3).cuda(), mode='replicate')

        m = M()
        [m] = easier.compile([m], backend='none')

        assert isinstance(m.v, easier.Tensor)
        assert isinstance(m.e, easier.Tensor)
        assert isinstance(m.r, easier.Tensor)

        cuda0 = torch.device('cuda:0')
        assert m.v.device == cuda0
        assert m.e.device == cuda0
        assert m.r.device == cuda0
        assert m.reducer.idx.device == cuda0

    @pytest.mark.parametrize('dev_type', [
        'cpu',
        pytest.param('cuda', marks=when_ngpus_ge_2)
    ])
    def test_collect_save(self, dev_type: str):
        torchrun_singlenode(
            1, worker__test_collect_save, (dev_type,),
            init_type=dev_type  # type: ignore
        )
