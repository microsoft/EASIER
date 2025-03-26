# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Union
import pytest
import torch
import easier


def fully_load_data(t: easier.Tensor):
    assert not t.easier_data_ready
    t.data = t.easier_data_loader.fully_load(None)
    t.easier_data_ready = True


def fully_load_idx(m: Union[easier.Selector, easier.Reducer]):
    assert m.easier_index_status == 'placeholder'
    m.idx = m.easier_data_loader.fully_load(None)
    m.easier_index_status = 'rewritten'


class TestSelector:
    idx = torch.randint(0, 10, (20,))
    selector = easier.Selector(idx)
    fully_load_idx(selector)

    def test_selector(self):
        t = torch.randn((10, 1, 2))
        edge = self.selector(t)

        assert torch.equal(edge, t[self.idx])


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


class TestJitNoneBackendUsage:
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

    def test_collect(self):
        n = 3

        class M(easier.Module):
            def __init__(self) -> None:
                super().__init__()
                self.v = easier.Tensor(
                    torch.rand(3, 3, dtype=torch.float64), mode='partition')
                self.e = easier.Tensor(
                    torch.rand(4, 3, dtype=torch.float64), mode='partition')
                self.r = easier.Tensor(
                    torch.rand(9, 3, dtype=torch.float64), mode='replicate')

        m = M()
        [m] = easier.compile([m], backend='none')

        assert torch.equal(m.v.collect(), m.v)
        assert torch.equal(m.e.collect(), m.e)
        assert torch.equal(m.r.collect(), m.r)
