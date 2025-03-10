# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing_extensions import Literal
from unittest.mock import patch
import torch
from torch.fx import GraphModule
import pytest

import h5py
import tempfile
import os

from easier.core.runtime.data_loader import \
    DataLoaderBase, InMemoryTensorLoader, H5DataLoader, FulledTensorLoader, \
    ArangeTensorLoader

from ..utils import torchrun_singlenode, get_random_str


def get_in_memory_tensor_loader(
    dtype: torch.dtype, device_type: Literal['cpu', 'cuda']
):
    v = torch.arange(17) * 3 + 1
    v = v.to(dtype=dtype, device=device_type)  # e.g. 'cuda' equals 'cuda:0'
    return InMemoryTensorLoader(v)


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


def worker__test_load_by_rank(local_rank: int, world_size: int,
                              data_loader_ctor, dtype: torch.dtype,
                              device_type: str):
    dl: DataLoaderBase = data_loader_ctor(dtype, device_type)
    assert dl.dtype == dtype
    assert dl.user_device.type == device_type
    assert dl.shape == (17,)

    tensor, start, end = dl.partially_load_by_rank()

    assert tensor.dtype == dtype
    assert tensor.device.type == 'cpu'  # by rank always CPU

    assert start == local_rank * 8
    assert end == (8 if local_rank == 0 else 17)
    length = end - start
    assert torch.equal(torch.arange(0, length, dtype=dtype)
                       * 3 + 1 + local_rank * 24, tensor)


def worker__test_load_by_index(local_rank: int, world_size: int,
                               data_loader_ctor, dtype: torch.dtype,
                               device_type: str):
    dl: DataLoaderBase = data_loader_ctor(dtype, device_type)
    assert dl.dtype == dtype
    assert dl.user_device.type == device_type
    assert dl.shape == (17,)

    idx = torch.arange(5) * 2 + local_rank
    tensor = dl.partially_load_by_index(idx, chunk_size=3)

    assert tensor.dtype == dtype
    assert tensor.device.type == 'cpu'  # by rank always CPU

    assert torch.equal(torch.arange(5, dtype=dtype) *
                       6 + 1 + local_rank * 3, tensor)


have_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")


@pytest.mark.parametrize('data_loader_ctor',
                         [get_in_memory_tensor_loader, get_h5_tensor_loader])
@pytest.mark.parametrize('dtype',
                         [torch.int64, torch.float64], ids=['i64', 'f64'])
@pytest.mark.parametrize('device_type', [
    'cpu',
    # no device IDs, all workers use cuda:0.
    pytest.param('cuda', marks=have_cuda)
])
class TestDataLoader:
    def test_load_chunk(self, data_loader_ctor, dtype: torch.dtype,
                        device_type: str):
        # rank-0 only
        dl: DataLoaderBase = data_loader_ctor(dtype, device_type)
        assert dl.dtype == dtype
        assert dl.user_device.type == device_type
        assert dl.shape == (17,)

        it = dl.partially_load_by_chunk(7)
        chunks = list(it)

        for chunk in chunks:
            assert chunk.dtype == dtype
            assert chunk.device == torch.device('cpu')

        assert torch.equal(torch.arange(0, 7, dtype=dtype) * 3 + 1, chunks[0])
        assert torch.equal(torch.arange(7, 14, dtype=dtype) * 3 + 1, chunks[1])
        assert torch.equal(torch.arange(
            14, 17, dtype=dtype) * 3 + 1, chunks[2])

    def test_load_by_rank(self, data_loader_ctor, dtype: torch.dtype,
                          device_type: str):
        torchrun_singlenode(2, worker__test_load_by_rank,
                          (data_loader_ctor, dtype, device_type))

    def test_load_by_index(self, data_loader_ctor, dtype: torch.dtype,
                           device_type: str):
        torchrun_singlenode(2, worker__test_load_by_index,
                          (data_loader_ctor, dtype, device_type))

    @pytest.mark.parametrize('target_device_type', [
        None,
        'cpu',
        pytest.param(
            'cuda', marks=have_cuda)
    ])
    def test_fully_load(self, data_loader_ctor, dtype: torch.dtype,
                        device_type: str, target_device_type: str):
        dl: DataLoaderBase = data_loader_ctor(dtype, device_type)
        assert dl.dtype == dtype
        assert dl.user_device.type == device_type
        assert dl.shape == (17,)

        tensor = dl.fully_load(target_device_type)
        assert tensor.dtype == dtype
        if target_device_type is None:
            assert tensor.device.type == device_type
        else:
            assert tensor.device.type == target_device_type


def worker__test_load_full_by_rank(local_rank: int, world_size: int,
                                   dtype: torch.dtype, device_type: str):
    dl = FulledTensorLoader(
        42, shape=[17, 2], dtype=dtype, device=torch.device(device_type))
    assert dl.dtype == dtype
    assert dl.user_device.type == device_type
    assert dl.shape == (17, 2)

    tensor, start, end = dl.partially_load_by_rank()

    assert tensor.dtype == dtype
    assert tensor.device.type == 'cpu'  # by rank always CPU

    assert start == local_rank * 8
    assert end == (8 if local_rank == 0 else 17)
    length = end - start
    assert torch.equal(torch.full([length, 2], 42, dtype=dtype), tensor)


def worker__test_load_full_by_index(local_rank: int, world_size: int,
                                    dtype: torch.dtype, device_type: str):

    dl = FulledTensorLoader(
        42, shape=[17, 2], dtype=dtype, device=torch.device(device_type))
    assert dl.dtype == dtype
    assert dl.user_device.type == device_type
    assert dl.shape == (17, 2)

    idx = torch.arange(5) * 2 + local_rank
    tensor = dl.partially_load_by_index(idx)

    assert tensor.dtype == dtype
    assert tensor.device.type == 'cpu'  # by rank always CPU

    assert torch.equal(torch.full([5, 2], 42, dtype=dtype), tensor)


@pytest.mark.parametrize('dtype',
                         [torch.int64, torch.float64], ids=['i64', 'f64'])
@pytest.mark.parametrize('device_type', [
    'cpu',
    # no device IDs, all workers use cuda:0.
    pytest.param('cuda', marks=have_cuda)
])
class TestFulledLoader:
    def test_load_chunk(self, dtype: torch.dtype, device_type: str):
        dl = FulledTensorLoader(
            42, shape=[17, 2], dtype=dtype, device=torch.device(device_type))
        assert dl.dtype == dtype
        assert dl.user_device.type == device_type
        assert dl.shape == (17, 2)

        it = dl.partially_load_by_chunk(7)
        chunks = list(it)

        for chunk in chunks:
            assert chunk.dtype == dtype
            assert chunk.device == torch.device('cpu')

        assert torch.equal(torch.full([7, 2], 42, dtype=dtype), chunks[0])
        assert torch.equal(torch.full([7, 2], 42, dtype=dtype), chunks[1])
        assert torch.equal(torch.full([3, 2], 42, dtype=dtype), chunks[2])

    def test_load_by_rank(self, dtype: torch.dtype, device_type: str):
        torchrun_singlenode(2, worker__test_load_full_by_rank,
                          (dtype, device_type))

    def test_load_by_index(self, dtype: torch.dtype, device_type: str):
        torchrun_singlenode(2, worker__test_load_full_by_index,
                          (dtype, device_type))


def worker__test_load_arange_by_rank(local_rank: int, world_size: int,
                                     dtype: torch.dtype, device_type: str):
    dl = ArangeTensorLoader(0, 34, 2,
                            dtype=dtype, device=torch.device(device_type))
    assert dl.dtype == dtype
    assert dl.user_device.type == device_type
    assert dl.shape == (17,)

    tensor, start, end = dl.partially_load_by_rank()

    assert tensor.dtype == dtype
    assert tensor.device.type == 'cpu'  # by rank always CPU

    assert start == local_rank * 8
    assert end == (8 if local_rank == 0 else 17)
    assert torch.equal(torch.arange(
        start * 2, end * 2, 2, dtype=dtype), tensor)


def worker__test_load_arange_by_index(local_rank: int, world_size: int,
                                      dtype: torch.dtype, device_type: str):

    dl = ArangeTensorLoader(0, 34, 2,
                            dtype=dtype, device=torch.device(device_type))
    assert dl.dtype == dtype
    assert dl.user_device.type == device_type
    assert dl.shape == (17,)

    idx = torch.arange(5) * 2 + local_rank
    tensor = dl.partially_load_by_index(idx)

    assert tensor.dtype == dtype
    assert tensor.device.type == 'cpu'  # by rank always CPU

    assert torch.equal(idx.to(dtype) * 2, tensor)


@pytest.mark.parametrize('dtype',
                         [torch.int64, torch.float64], ids=['i64', 'f64'])
@pytest.mark.parametrize('device_type', [
    'cpu',
    # no device IDs, all workers use cuda:0.
    pytest.param('cuda', marks=have_cuda)
])
class TestArangeLoader:
    def test_load_chunk(self, dtype: torch.dtype, device_type: str):
        dl = ArangeTensorLoader(0, 34, 2,
                                dtype=dtype, device=torch.device(device_type))
        assert dl.dtype == dtype
        assert dl.user_device.type == device_type
        assert dl.shape == (17,)

        it = dl.partially_load_by_chunk(7)
        chunks = list(it)

        for chunk in chunks:
            assert chunk.dtype == dtype
            assert chunk.device == torch.device('cpu')

        assert torch.equal(torch.arange(0, 14, 2, dtype=dtype), chunks[0])
        assert torch.equal(torch.arange(14, 28, 2, dtype=dtype), chunks[1])
        assert torch.equal(torch.arange(28, 34, 2, dtype=dtype), chunks[2])

    def test_load_by_rank(self, dtype: torch.dtype, device_type: str):
        torchrun_singlenode(2, worker__test_load_arange_by_rank,
                          (dtype, device_type))

    def test_load_by_index(self, dtype: torch.dtype, device_type: str):
        torchrun_singlenode(2, worker__test_load_arange_by_index,
                          (dtype, device_type))
