# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from unittest.mock import patch
from typing_extensions import Literal
import torch
import pytest

import h5py
import tempfile
import os

import easier
from easier.core.runtime.data_loader.factories import \
    DataLoaderBase, InMemoryTensorLoader, H5DataLoader, ArangeDataLoader
from easier.core.runtime.data_loader.ops import \
    CartesianProductDataLoader, ConcatDataLoader
from easier.core.runtime.data_loader.utils import \
    NormalizedSlice, get_overlapping_slice, CopyingSlicer, get_strides

from easier.core.runtime.dist_env import get_default_dist_env
from easier.core.utils import get_random_str
from tests.utils import \
    torchrun_singlenode, have_cuda, when_ngpus_ge_2, torchrun_spawn


def vec(*vs, dtype=torch.int64):
    return torch.tensor(vs, dtype=dtype)


class TestNormalizedSlice:
    def test_from_to_slice(self):
        length, start, stop, step = 1, 0, 0, 1
        ns = NormalizedSlice.from_slice(length, slice(start, stop, step))
        assert ns == NormalizedSlice(length, 0, 1, 0)
        assert ns.to_slice() == slice(0, 0, 1)

        length, start, stop, step = 10, 0, -3, 2
        ns = NormalizedSlice.from_slice(length, slice(start, stop, step))
        assert ns == NormalizedSlice(length, 0, 2, 4)
        assert ns.to_slice() == slice(0, 8, 2)

        length, start, stop, step = 10, -1, 2, -2
        ns = NormalizedSlice.from_slice(length, slice(start, stop, step))
        assert ns == NormalizedSlice(length, 9, -2, 4)
        assert ns.to_slice() == slice(9, 1, -2)

        length, start, stop, step = 10, -1, -20, -2
        ns = NormalizedSlice.from_slice(length, slice(start, stop, step))
        assert ns == NormalizedSlice(length, 9, -2, 5)
        assert ns.to_slice() == slice(9, None, -2)

        length, start, stop, step = 10, -2, -20, -2
        ns = NormalizedSlice.from_slice(length, slice(start, stop, step))
        assert ns == NormalizedSlice(length, 8, -2, 5)
        assert ns.to_slice() == slice(8, None, -2)

    def test_compose(self):
        ns1 = NormalizedSlice(20, 1, 2, 7)

        with pytest.raises(ValueError, match='maintain.*shapes'):
            ns1.compose(ns1)

        assert ns1.compose(
            NormalizedSlice(7, 1, 2, 3)
        ) == NormalizedSlice(20, 3, 4, 3)

        assert ns1.compose(
            NormalizedSlice(7, 6, -3, 2)
        ) == NormalizedSlice(20, 13, -6, 2)

        ns2 = NormalizedSlice(20, 18, -3, 6)

        assert ns2.compose(
            NormalizedSlice(6, 1, 2, 3)
        ) == NormalizedSlice(20, 15, -6, 3)

        assert ns2.compose(
            NormalizedSlice(6, 5, -1, 4)
        ) == NormalizedSlice(20, 3, 3, 4)

    def test_split(self):
        ns1 = NormalizedSlice(200, 3, 5, 30)
        assert ns1.split(8) == [
            NormalizedSlice(200, 3, 5, 8),
            NormalizedSlice(200, 43, 5, 8),
            NormalizedSlice(200, 83, 5, 8),
            NormalizedSlice(200, 123, 5, 6),
        ]

        ns2 = NormalizedSlice(200, 180, -5, 30)
        assert ns2.split(8) == [
            NormalizedSlice(200, 180, -5, 8),
            NormalizedSlice(200, 140, -5, 8),
            NormalizedSlice(200, 100, -5, 8),
            NormalizedSlice(200, 60, -5, 6),
        ]

    def test_reverse(self):
        ns1 = NormalizedSlice(200, 3, 5, 30)
        assert ns1.reverse() == NormalizedSlice(200, 148, -5, 30)

        ns2 = NormalizedSlice(200, 180, -5, 30)
        assert ns2.reverse() == NormalizedSlice(200, 35, 5, 30)

    def test_overlap_region_directionless(self):
        regions = [
            NormalizedSlice(100, 0, 1, 100),
            NormalizedSlice(100, 99, -1, 100),
        ]
        selections = [
            NormalizedSlice(100, 22, 5, 8),
            NormalizedSlice(100, 88, -5, 8),
        ]

        for r in regions:
            for s in selections:
                assert get_overlapping_slice(r, s) == s

    def test_overlap(self):
        assert get_overlapping_slice(
            NormalizedSlice(100, 7, 3, 30), NormalizedSlice(100, 11, 5, 15)
        ) == NormalizedSlice(100, 16, 15, 5)

        assert get_overlapping_slice(
            NormalizedSlice(100, 7, 3, 30), NormalizedSlice(100, 97, -4, 20)
        ) == NormalizedSlice(100, 85, -12, 6)


def get_h5_tensor_loader(dtype: torch.dtype,
                         device_type: Literal['cpu', 'cuda']):
    fn = get_random_str() + ".hdf5"
    dir = os.path.join(tempfile.gettempdir(), "easier", "tests")
    os.makedirs(dir, exist_ok=True)

    v = torch.arange(17) * 3 + 1

    fpath = os.path.join(dir, fn)
    with h5py.File(fpath, 'w') as f:
        f.create_dataset("d", data=v.numpy())

    return H5DataLoader(fpath, "d", dtype=dtype, device=device_type)


def get_in_memory_tensor_loader(
    dtype: torch.dtype, device_type: Literal['cpu', 'cuda']
):
    v = torch.arange(17) * 3 + 1
    v = v.to(dtype=dtype, device=device_type)  # e.g. 'cuda' equals 'cuda:0'
    return InMemoryTensorLoader(v)


class TestDataLoaderBase:
    @pytest.mark.parametrize('data_loader_ctor',
                             [get_in_memory_tensor_loader, get_h5_tensor_loader])
    @pytest.mark.parametrize('dtype',
                             [torch.int64, torch.float64], ids=['i64', 'f64'])
    @pytest.mark.parametrize('device_type', [
        'cpu',
        # no device IDs, all workers use cuda:0.
        pytest.param('cuda', marks=have_cuda)
    ])
    @pytest.mark.usefixtures('dummy_dist_env')
    def test_to(self, data_loader_ctor, dtype: torch.dtype, device_type: str):
        dl: DataLoaderBase = data_loader_ctor(dtype, 'cpu')

        if dtype.is_floating_point:
            dl_f16 = dl.to(dtype=torch.float16)
            assert dl is not dl_f16
            assert dl_f16.device.type == 'cpu'
            assert dl_f16.dtype == torch.float16
        else:
            dl_i8 = dl.to(dtype=torch.int8)
            assert dl is not dl_i8
            assert dl_i8.device.type == 'cpu'
            assert dl_i8.dtype == torch.int8  # not changed

        dl_device = dl.to(device=device_type)
        assert dl is not dl_device
        assert dl_device.device.type == device_type
        assert dl_device.dtype == dtype

    @pytest.mark.usefixtures('dummy_dist_env')
    def test_chunk(self):
        dl = ArangeDataLoader(3, 3, 33, torch.float64, 'cpu')
        l = list(dl._load_by_chunk_rank0(
            NormalizedSlice(33, 0, 1, 33),
            chunk_size=10
        ))
        assert len(l) == 4
        assert torch.equal(torch.arange(3, 31, 3, dtype=torch.float64), l[0])
        assert torch.equal(torch.arange(93, 100, 3, dtype=torch.float64), l[3])

        l = list(dl._load_by_chunk_rank0(
            NormalizedSlice(33, 32, -2, 16),
            chunk_size=10
        ))
        assert len(l) == 2
        assert torch.equal(torch.arange(99, 44, -6, dtype=torch.float64), l[0])
        assert torch.equal(torch.arange(39, 8, -6, dtype=torch.float64), l[1])

    @pytest.mark.usefixtures('dummy_dist_env')
    def test_minmax_unique(self):
        with patch(
            f'{DataLoaderBase.__module__}.{DataLoaderBase.__name__}'
                '.CHUNK_SIZE',
            new=10,
        ), patch(
            f'{DataLoaderBase.__module__}.{DataLoaderBase.__name__}'
                '.BITPACK_MAXLEN',
            new=10
        ):
            dl = ArangeDataLoader(3, 3, 33, torch.int64, 'cpu')
            amin, amax = dl.minmax(NormalizedSlice(33, 0, 1, 33))
            assert (amin, amax) == (3, 99)
            nunique = dl.count_unique(NormalizedSlice(33, 0, 1, 33))
            assert nunique == 33

            amin, amax = dl.minmax(NormalizedSlice(33, 32, -2, 16))
            assert (amin, amax) == (9, 99)
            nunique = dl.count_unique(NormalizedSlice(33, 32, -2, 16))
            assert nunique == 16

    @pytest.mark.usefixtures('dummy_dist_env')
    def test_getitem(self):
        t = torch.rand(5, 4, 3)
        dl = InMemoryTensorLoader(t)

        view = dl[1]
        data = view.fully_load(torch.device('cpu'), True)
        assert view.shape == (4, 3)
        assert torch.equal(data, t[1])

        view = dl[1:]
        data = view.fully_load(torch.device('cpu'), True)
        assert view.shape == (4, 4, 3)
        assert torch.equal(data, t[1:])

        view = dl[-1:-3:-1, 2, :]
        data = view.fully_load(torch.device('cpu'), True)
        assert view.shape == (2, 3)
        assert torch.equal(data, CopyingSlicer(t)[-1:-3:-1, 2, :])


@pytest.mark.parametrize(
    'dtype', [torch.int64, torch.float64], ids=['i64', 'f64']
)
class TestH5DataLoader:
    def worker__test_partial_load(
        self, local_rank: int, world_size: int, dtype: torch.dtype
    ):
        # only the data file on rank-0 is effective
        dl = get_h5_tensor_loader(dtype, 'cpu')
        assert dl.dtype == dtype
        assert dl.device.type == 'cpu'
        assert dl.shape == (17,)

        tensor = dl.partially_load_by_range(
            NormalizedSlice(17, local_rank, 3, 5)
        )
        assert tensor.dtype == dtype
        assert tensor.device.type == 'cpu'  # by rank always CPU
        assert torch.equal(
            torch.arange(5, dtype=dtype) * 9 + 1 + local_rank * 3, tensor
        )

        tensor = dl.partially_load_by_range(
            NormalizedSlice(17, 16 - local_rank, -3, 5)
        )
        assert tensor.dtype == dtype
        assert tensor.device.type == 'cpu'  # by rank always CPU
        assert torch.equal(
            49 - torch.arange(5, dtype=dtype) * 9 - local_rank * 3, tensor
        )

        idx = vec(1, 3, 9, 7, 5, 15, 11, 13) + local_rank
        tensor = dl.partially_load_by_index(idx)
        assert tensor.dtype == dtype
        assert tensor.device.type == 'cpu'  # by rank always CPU
        assert torch.equal(idx.to(dtype) * 3 + 1, tensor)

    def test_partial_load(self, dtype: torch.dtype):
        torchrun_singlenode(
            2, self.worker__test_partial_load, (dtype,)
        )

    def worker__test_fully_load(
        self, local_rank: int, world_size: int,
        dtype: torch.dtype, device_type: str, final_device_type: str
    ):
        dl = get_h5_tensor_loader(dtype, device_type)  # type: ignore

        final_device = torch.device(final_device_type, local_rank)

        v = torch.arange(17) * 3 + 1
        v = v.to(dtype=dtype, device=final_device)

        tensor = dl.fully_load(final_device, replicated=False)

        assert tensor.dtype == dtype
        if final_device_type == 'cpu':
            assert tensor.device.type == 'cpu'
        else:
            assert tensor.device == final_device

        if local_rank == 0:
            assert torch.equal(v, tensor)
        else:
            assert not torch.equal(v, tensor)

        tensor = dl.fully_load(final_device, replicated=True)
        assert tensor.dtype == dtype
        if final_device_type == 'cpu':
            assert tensor.device.type == 'cpu'
        else:
            assert tensor.device == final_device
        assert torch.equal(v, tensor)

    @pytest.mark.parametrize('device_type', [
        'cpu',
        # no device IDs, all workers use cuda:0.
        pytest.param('cuda', marks=have_cuda)
    ])
    @pytest.mark.parametrize('final_device_type', [
        'cpu',
        pytest.param('cuda', marks=when_ngpus_ge_2)
    ])
    def test_fully_load(
        self, dtype: torch.dtype, device_type: str, final_device_type: str
    ):
        torchrun_singlenode(
            2, self.worker__test_fully_load,
            (dtype, device_type, final_device_type)
        )


@pytest.mark.usefixtures('dummy_dist_env')
class TestArangeDataLoader:
    def _test(self, t: torch.Tensor, dl: DataLoaderBase, eq):
        l = t.shape[0]

        t2 = dl.fully_load(torch.device('cpu'), True)
        assert t.shape == t2.shape
        assert eq(t, t2)

        t2 = dl.partially_load_by_range(NormalizedSlice(l, 0, 1, l))
        assert t.shape == t2.shape
        assert eq(t, t2)

        ns = NormalizedSlice(l, l // 3, 2, l // 3)
        t1 = CopyingSlicer(t)[ns.to_slice()]
        t2 = dl.partially_load_by_range(ns)
        assert eq(t1, t2)

        ns = NormalizedSlice(l, l // 3, 2, l // 3).reverse()
        t1 = CopyingSlicer(t)[ns.to_slice()]
        t2 = dl.partially_load_by_range(ns)
        assert eq(t1, t2)

        idx = vec(1, 3, 7, 5)
        t1 = t[idx]
        t2 = dl.partially_load_by_index(idx)
        assert eq(t1, t2)

    def test_int(self):
        for (start, step, count) in [
            (1, 1, 10),
            (2, 7, 20),
            (99, -11, 9),
        ]:
            stop = start + step * count
            t = torch.arange(start, stop, step, dtype=torch.int64)
            dl = ArangeDataLoader(start, step, count, torch.int64, 'cpu')
            self._test(t, dl, torch.equal)

    def test_double(self):
        for (start, step, count) in [
            (1, 1, 10),
            (2, 7, 20),
            (99, -11, 9),
            (1, 0.1, 30),
            (1.5, 2.4, 17),
            (21.5, -2.4, 10),
        ]:
            stop = start + step * count
            t = torch.arange(start, stop, step, dtype=torch.float64)
            dl = ArangeDataLoader(start, step, count, torch.float64, 'cpu')
            self._test(t, dl, torch.allclose)

    def test_arange__overloading(self):
        assert easier.arange(10).shape == (10,)
        assert easier.arange(5, 10).shape == (5,)
        assert easier.arange(5, 10, 2).shape == (3,)

    def test_linspace(self):
        import numpy
        for (start, stop, num, endpoint) in [
            (1, 2, 10, True),
            (1, 2, 10, False),
            (-5, -2, 10, True),
            (-5, -2, 10, False),
        ]:
            t = torch.from_numpy(numpy.linspace(
                start, stop, num, endpoint=endpoint, dtype=numpy.float64
            ))
            dl = easier.linspace(
                start, stop, num, endpoint=endpoint, dtype=torch.float64
            )
            self._test(t, dl, torch.allclose)

    def test_linspace__special_cases(self):
        import numpy
        for (start, stop, num, endpoint) in [
            (1, 2, 0, True),
            (1, 2, 0, False),
            (1, 2, 1, True),
            (1, 2, 1, False),
        ]:
            t = torch.from_numpy(numpy.linspace(
                start, stop, num, endpoint=endpoint, dtype=numpy.float64
            ))
            dl = easier.linspace(
                start, stop, num, endpoint=endpoint, dtype=torch.float64
            )

            t2 = dl.fully_load(torch.device('cpu'), True)
            assert t.shape == t2.shape
            assert torch.equal(t, t2)

    def test_linspace__stop_rounding(self):
        import numpy
        for (start, stop, nintervals) in [
            (0, 1, 107)
        ]:
            assert (stop - start) / nintervals * nintervals != (stop - start)

            num = nintervals + 1
            t = torch.from_numpy(numpy.linspace(
                start, stop, num, endpoint=True, dtype=numpy.float64
            ))
            dl = easier.linspace(
                start, stop, num, endpoint=True, dtype=torch.float64
            )
            self._test(t, dl, torch.allclose)

            t2 = dl.fully_load(torch.device('cpu'), True)
            assert t2[0] == start
            assert t2[-1] == stop


class TestStridedDataLoader:
    @torchrun_spawn()
    def worker__test_dim0_int(self):
        _m = torch.arange(42).reshape(3, 2, 1, 7)
        m = CopyingSlicer(_m)[2, 1, 0, 6:1:-2]

        dl = InMemoryTensorLoader(_m)
        sdl = dl[2, 1, 0, 6:1:-2]

        assert sdl.shape == (3,)

        ns1 = NormalizedSlice(3, 0, 1, 3)
        assert torch.equal(
            m, sdl.partially_load_by_range(ns1)
        )
        assert sdl.minmax(ns1) == (37, 41)
        assert sdl.count_unique(ns1) == 3

        ns2 = NormalizedSlice(3, 2, -1, 2)
        assert torch.equal(
            m[[2, 1]], sdl.partially_load_by_range(ns2)
        )
        assert sdl.minmax(ns2) == (37, 39)
        assert sdl.count_unique(ns2) == 2

        assert torch.equal(
            m, sdl.fully_load(torch.device('cpu'), True)
        )

    @torchrun_spawn()
    def worker__test_basic(self):
        _m = torch.arange(35).reshape(5, 7)
        m = torch.from_numpy(_m.numpy()[4::-2, 5::-2].copy())

        dl = InMemoryTensorLoader(_m)
        sdl = dl[4::-2, 5::-2]

        assert sdl.shape == (3, 3)

        ns1 = NormalizedSlice(3, 0, 1, 3)
        assert torch.equal(
            m, sdl.partially_load_by_range(ns1)
        )
        assert sdl.minmax(ns1) == (1, 33)
        assert sdl.count_unique(ns1) == 9

        ns2 = NormalizedSlice(3, 2, -1, 2)
        assert torch.equal(
            m[[2, 1]], sdl.partially_load_by_range(ns2)
        )
        assert sdl.minmax(ns2) == (1, 19)
        assert sdl.count_unique(ns2) == 6

        assert torch.equal(
            m, sdl.fully_load(torch.device('cpu'), True)
        )


class TestCartesianProductDataLoader:
    @torchrun_spawn()
    def worker__test(self):
        cdl = CartesianProductDataLoader([
            easier.arange(0, 5),
            easier.arange(50, 57),
            easier.arange(100, 111),
        ])
        cdl._chunk_size = 42

        raw = torch.cartesian_prod(
            torch.arange(0, 5),
            torch.arange(50, 57),
            torch.arange(100, 111),
        )

        idx = torch.arange(3, 11, 2)
        assert torch.equal(cdl.partially_load_by_index(idx), raw[idx])

        idx = torch.arange(155, 162)
        assert torch.equal(cdl.partially_load_by_index(idx), raw[idx])

        idx = torch.arange(383, 200, -5)
        assert torch.equal(cdl.partially_load_by_index(idx), raw[idx])

        assert torch.equal(raw, cdl.fully_load(torch.device('cpu'), True))

        if get_default_dist_env().rank == 0:
            assert torch.equal(raw, cdl.fully_load(torch.device('cpu'), False))


class TestConcatDataLoader:
    @torchrun_spawn()
    def worker__test_ascend(self):
        cdl = ConcatDataLoader([
            easier.arange(0, 20),
            easier.arange(50, 80),
            easier.arange(100, 150),
        ])

        class _Ascend:
            def __init__(self) -> None:
                self.call = 0

            def __call__(self, comp, concat_overlap, comp_overlap):
                assert isinstance(comp, ArangeDataLoader)
                if self.call == 0:
                    assert comp._start == 0
                    assert concat_overlap == NormalizedSlice(100, 7, 3, 5)
                    assert comp_overlap == NormalizedSlice(20, 7, 3, 5)
                elif self.call == 1:
                    assert comp._start == 50
                    assert concat_overlap == NormalizedSlice(100, 22, 3, 10)
                    assert comp_overlap == NormalizedSlice(30, 2, 3, 10)
                elif self.call == 2:
                    assert comp._start == 100
                    assert concat_overlap == NormalizedSlice(100, 52, 3, 15)
                    assert comp_overlap == NormalizedSlice(50, 2, 3, 15)
                else:
                    assert False
                self.call += 1
        ns = NormalizedSlice(100, 7, 3, 30)
        cdl._foreach_component(ns, _Ascend())

        assert cdl.minmax(NormalizedSlice(100, 0, 1, 100)) == (0, 149)
        assert cdl.minmax(NormalizedSlice(100, 10, 1, 5)) == (10, 14)
        assert cdl.minmax(NormalizedSlice(100, 14, -1, 5)) == (10, 14)

        assert torch.equal(
            torch.concat([
                torch.arange(7, 20, 3),
                torch.arange(52, 80, 3),
                torch.arange(102, 145, 3),
            ]),
            cdl.partially_load_by_range(ns)
        )

        assert torch.equal(
            vec(10, 13, 60, 70, 120, 130),
            cdl.partially_load_by_index(vec(10, 13, 30, 40, 70, 80))
        )

        assert torch.equal(
            vec(10, 13, 60, 70, 120, 130).flip(0),
            cdl.partially_load_by_index(vec(10, 13, 30, 40, 70, 80).flip(0))
        )

        raw = torch.concat([
            torch.arange(0, 20), torch.arange(50, 80), torch.arange(100, 150)
        ])

        assert torch.equal(raw, cdl.fully_load(torch.device('cpu'), True))

        if get_default_dist_env().rank == 0:
            assert torch.equal(raw, cdl.fully_load(torch.device('cpu'), False))

    @torchrun_spawn()
    def worker__test_descend(self):
        cdl = ConcatDataLoader([
            easier.arange(0, 20),
            easier.arange(50, 80),
            easier.arange(100, 150),
        ])

        class _Descend:
            def __init__(self) -> None:
                self.call = 0

            def __call__(self, comp, concat_overlap, comp_overlap):
                if self.call == 0:
                    assert comp._start == 100
                    assert concat_overlap == NormalizedSlice(100, 91, -3, 14)
                    assert comp_overlap == NormalizedSlice(50, 41, -3, 14)
                elif self.call == 1:
                    assert comp._start == 50
                    assert concat_overlap == NormalizedSlice(100, 49, -3, 10)
                    assert comp_overlap == NormalizedSlice(30, 29, -3, 10)
                elif self.call == 2:
                    assert comp._start == 0
                    assert concat_overlap == NormalizedSlice(100, 19, -3, 6)
                    assert comp_overlap == NormalizedSlice(20, 19, -3, 6)
                else:
                    assert False
                self.call += 1
        ns = NormalizedSlice(100, 91, -3, 30)
        cdl._foreach_component(ns, _Descend())

        assert torch.equal(
            torch.concat([
                torch.arange(141, 101, -3),
                torch.arange(79, 50, -3),
                torch.arange(19, 3, -3),
            ]),
            cdl.partially_load_by_range(ns)
        )


class TestMesh:
    class _IndicesTester:
        def __init__(self, mesh: easier.Mesh, full_coords_dims: list) -> None:
            self.mesh = mesh
            self.full_coords_dims = full_coords_dims

        def __getitem__(self, indices: tuple):
            idxdl = self.mesh.indices[*indices]

            coords_dims = []
            for full_coords_dim, index in zip(self.full_coords_dims, indices):
                coords_dim = CopyingSlicer(full_coords_dim)[index]
                if isinstance(index, int):
                    assert coords_dim.shape == ()
                    coords_dim = coords_dim[None]
                assert len(coords_dim.shape) == 1
                coords_dims.append(coords_dim)

            if len(indices) < len(self.full_coords_dims):
                coords_dims.extend(self.full_coords_dims[len(indices):])

            coords = torch.cartesian_prod(*coords_dims)

            assert idxdl.shape == (coords.shape[0],)
            idx = idxdl.fully_load(torch.device('cpu'), True)

            strides = get_strides(self.mesh._nvs)
            assert torch.equal(idx, (coords * strides).sum(dim=1))

    @pytest.mark.usefixtures('dummy_dist_env')
    def test_2d(self):
        mesh = easier.Mesh(
            easier.linspace(0, 1, 4),
            easier.linspace(0, 1, 5),
        )
        assert mesh.nv == 20
        assert mesh.ne == (3*4)*(2)*2 - (3 + 4)*2

        v = mesh.vertices.fully_load(torch.device('cpu'), True)
        src = mesh.src.fully_load(torch.device('cpu'), True)
        dst = mesh.dst.fully_load(torch.device('cpu'), True)

        assert torch.allclose(
            v,
            torch.cartesian_prod(
                torch.linspace(0, 1, 4),
                torch.linspace(0, 1, 5),
            ).to(torch.float64)
        )

        m = torch.arange(12).reshape(3, 4)

        assert torch.equal(
            src,
            torch.concat([
                m[:-1, :].flatten(),
                m[1:, :].flatten(),

                m[:, :-1].flatten(),
                m[:, 1:].flatten(),
            ], dim=0)
        )

        assert torch.equal(
            dst,
            torch.concat([
                m[1:, :].flatten(),
                m[:-1, :].flatten(),

                m[:, 1:].flatten(),
                m[:, :-1].flatten(),
            ], dim=0)
        )

        tester = TestMesh._IndicesTester(mesh, [
            torch.arange(4),
            torch.arange(5)
        ])

        tester[:, :]
        tester[1:4:2, :-1]
        tester[3, 3]
        tester[-2, -2]

    @pytest.mark.usefixtures('dummy_dist_env')
    def test_3d(self):
        mesh = easier.Mesh(
            easier.linspace(0, 1, 4),
            easier.linspace(0, 1, 5),
            easier.linspace(0, 1, 6),
        )
        assert mesh.nv == 120
        assert mesh.ne == (3*4*5)*(3)*2 - (4*5 + 3*5 + 3*4)*2

        v = mesh.vertices.fully_load(torch.device('cpu'), True)
        src = mesh.src.fully_load(torch.device('cpu'), True)
        dst = mesh.dst.fully_load(torch.device('cpu'), True)

        assert torch.allclose(
            v,
            torch.cartesian_prod(
                torch.linspace(0, 1, 4),
                torch.linspace(0, 1, 5),
                torch.linspace(0, 1, 6),
            ).to(torch.float64)
        )

        m = torch.arange(60).reshape(3, 4, 5)

        assert torch.equal(
            src,
            torch.concat([
                m[:-1, :, :].flatten(),
                m[1:, :, :].flatten(),

                m[:, :-1, :].flatten(),
                m[:, 1:, :].flatten(),

                m[:, :, :-1].flatten(),
                m[:, :, 1:].flatten(),
            ], dim=0)
        )

        assert torch.equal(
            dst,
            torch.concat([
                m[1:, :, :].flatten(),
                m[:-1, :, :].flatten(),

                m[:, 1:, :].flatten(),
                m[:, :-1, :].flatten(),

                m[:, :, 1:].flatten(),
                m[:, :, :-1].flatten(),
            ], dim=0)
        )

        tester = TestMesh._IndicesTester(mesh, [
            torch.arange(4),
            torch.arange(5),
            torch.arange(6),
        ])

        tester[:, :, :]
        tester[1:4:2, :-1]
        tester[:, 1, -1]
        tester[-1, -1:0:-1]
