# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from contextlib import contextmanager
import os
from typing import Iterator, Optional, Tuple, Union, cast
import h5py

import numpy as np
import torch

from easier.core.runtime.dist_env import get_runtime_dist_env
from easier.core.runtime.utils import check_collective_equality


ATTRIBUTE_PLACEHOLDER = "easier_placeholder"


def _get_offset_exactly_nparts(orig_len: int, nparts: int, part: int
                               ) -> Tuple[int, int]:
    """
    A part will have roughly `orig_len // nparts` elements.
    The remaining elements will be put in the last part.

    Please note how the remaining elements are handled. When the remaining
    elements are treated as an individial part, this auxiliary method cannot
    be used.
    """
    per_worker_len = orig_len // nparts

    start = per_worker_len * part

    if part + 1 == nparts:
        end = orig_len
    else:
        end = per_worker_len * (part + 1)

    return start, end


def _check_dtype_shape_and_more(dt: 'DataLoaderBase', hint_name, *more_metas):
    dt_metas = [dt.dtype, dt.shape] + list(more_metas)
    check_collective_equality(f"Tensor properties of {hint_name}", dt_metas)


class DataLoaderBase:
    """
    The data loader for one specified data source, e.g. a HDF5 dataset.

    Calls to every method should be collective.
    """

    def __init__(self) -> None:
        """
        The constructor should do simple member data storage and local tasks.
        No collective calls should be done during construction.
        """
        self.shape: Tuple[int, ...]
        self.dtype: torch.dtype

        # The device on which the data loader is intially defined.
        # This device configuration only take effect with "torch" JIT backend.
        self.device: torch.device

    def collective_init(self, hint_name) -> None:
        """
        Validate if the the data of this data loader is collectively correct.

        Require callers to first ensure the data loders among workers are
        actually referring to the same data set.

        Args:
        - hint_name:
            e.g. "the data loader of (Module.a.b.c:Selector)"
            because spmd_init is not run when the data loader is constructed,
            we offer a clear hint name for users to locate the issue reported.
        """
        raise NotImplementedError()
    
    def minmax(self) -> Tuple[object, object]:
        raise NotImplementedError()
    
    def count_unique(self) -> int:
        """
        Used by Reducer.set_fullness()
        """
        raise NotImplementedError()
    
    # TODO now we accept keyword parameters only, we may make it do
    # overloading resolution like Selector(idx=)
    def to(
        self,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str]] = None
    ) -> 'DataLoaderBase':
        raise NotImplementedError()

    def partially_load_by_chunk(self, chunk_size: int
                                ) -> Iterator[torch.Tensor]:
        """
        Only callable at rank-0.

        Chuck size is only about the first dimension and item tensors in
        the resultant sequence:
        - always on CPU;
        - may not have batch size that exactly equals chunk_size.
        """
        raise NotImplementedError()

    def partially_load_by_rank(self) -> Tuple[torch.Tensor, int, int]:
        """
        Collectively load an evenly distributed part of the target dataset
        for each rank.

        Returns:
        - torch.Tensor: the loaded part, always on CPU
        - int: the beginning offset of the part (inclusive)
        - int: the end offset of the part (exclusive)
        """
        raise NotImplementedError()

    def partially_load_by_index(self, index: torch.Tensor, **kwargs
                                ) -> torch.Tensor:
        """
        Collectively load a part of the target dataset with the
        specified index tensor.
        The index is defined in the global index space.

        Args:
        - index: should always be on CPU
        - kwargs: subtype-specific config

        Returns:
        - torch.Tensor: the loaded part, always on CPU
        """
        raise NotImplementedError()

    def fully_load(self, device: Union[torch.device, str, None]
                   ) -> torch.Tensor:
        """
        When fully loading the dataset.

        Args:
        - device:
            If None, the initial device specified on the data loader is used;
            otherwise, load the data to the input device.

        Returns:
        - torch.Tensor: the full tensor, on the resolved device.
        """
        raise NotImplementedError()


    # def get_placeholder(self) -> torch.Tensor:
    #     """
    #     Allocate a new placeholder torch.Tensor of the same dtype/shape/device
    #     but generally consuming no memory to be compatible with cases where
    #     torch.Tensor object and information is needed.

    #     Any subclass implementation should add the attribute
    #     "easier_placeholder" to indicate the result is a placeholder, too.

    #     The placeholder is needed mainly to:
    #     1.  ease the inspection of tensor properties like dtype/shape/device,
    #         especially for the metadata pass.
    #     2.  fulfil `esr.Tensor.__new__(cls, data)` where the underlying data
    #         tensor should be set.
    #     """
    #     # torch.Tensor.expand can expand shape-(1,) to e.g. shape-(0,0,0),
    #     # but not to ndim=0 shape ().
    #     if len(self.shape) == 0:
    #         ph = torch.zeros((), dtype=self.dtype, device=self.device)
    #     else:
    #         ph = torch.zeros(
    #             (1,), dtype=self.dtype, device=self.device
    #         ).expand(self.shape)  # can even expand to (0,0,0)

    #     setattr(ph, ATTRIBUTE_PLACEHOLDER, True)
    #     return ph

    def __repr__(self) -> str:
        """
        When possible, return a string which could be treated as valid Python
        code to construct this data loader, except for `.device`, e.g.
        ```
        ArangeTensorLoader(start=0, end=1, step=1, dtype=torch.float64)
        ```

        NOTE this repr str will be used to validate compilation cache for
        mesh consistency across EASIER sessions.
        """
        raise NotImplementedError()


class InMemoryTensorLoader(DataLoaderBase):
    """
    Expected to have the same data on all ranks.

    Remarks:
    To initialize with empty data, do not use `torch.empty()`, use
    `torch.zeros()` instead.
    """

    def __init__(self, tensor: torch.Tensor) -> None:
        super().__init__()

        self.dtype = tensor.dtype
        self.shape = tensor.shape

        # The data is always stored as CPU tensor
        self.tensor = tensor.to('cpu')
    
    def collective_init(self, hint_name) -> None:
        _check_dtype_shape_and_more(self, hint_name)

        def _eq_tensor(v, v0):
            # torch.allclose support broadcasting, so we need to check shapes.
            return v.shape == v0.shape and torch.allclose(v, v0)
        check_collective_equality(
            f"The input tensor of {hint_name}", self.tensor, eq=_eq_tensor
        )

    def minmax(self) -> Tuple[object, object]:
        amin, amax = self.tensor.aminmax()
        return amin.item(), amax.item()

    def partially_load_by_chunk(self, chunk_size: int
                                ) -> Iterator[torch.Tensor]:
        cpu_dist_env = get_cpu_dist_env()
        assert cpu_dist_env.rank == 0, \
            "Loading-by-chunk is only available on rank-0"

        orig_len = self.tensor.shape[0]

        # Put tailing elements in an individual chunk whose size is smaller.
        nchunk, remainder = divmod(orig_len, chunk_size)
        if remainder > 0:
            nchunk += 1

        # After we have decided a valid nchunk (>=1), it won't matter even if
        # `_get_offset_exactly_nparts` to get offsets.
        # But we still follow the chunk partition above.
        for i in range(nchunk):
            start = chunk_size * i
            end = min(orig_len, chunk_size * (i + 1))
            chunk = self.tensor[start:end].clone()
            yield chunk

    def partially_load_by_rank(self) -> Tuple[torch.Tensor, int, int]:
        cpu_dist_env = get_cpu_dist_env()
        world_size = cpu_dist_env.world_size
        rank = cpu_dist_env.rank
        orig_len = self.tensor.shape[0]

        # Put tailing elements in the part for the last rank, making the size
        # of that part bigger than chunk_size.
        start, end = _get_offset_exactly_nparts(orig_len, world_size, rank)

        return self.tensor[start:end].clone(), start, end

    def partially_load_by_index(self, index: torch.Tensor,
                                **kwargs) -> torch.Tensor:
        assert index.device == torch.device('cpu')
        return self.tensor[index]

    def fully_load(self, device: Union[torch.device, str]) -> torch.Tensor:
        return self.tensor.to(device, copy=True)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(tensor={self.tensor})'


def numpy_dtype_to_torch_dtype(np_dtype: np.dtype):
    # torch.from_numpy accepts only:
    # float64, float32, float16, complex64, complex128,
    # int64, int32, int16, int8, uint8, bool
    # Otherwise it raises TypeError.
    return torch.from_numpy(np.ndarray(shape=[0], dtype=np_dtype)).dtype


def torch_dtype_to_numpy_dtype(torch_dtype: torch.dtype):
    return torch.empty([0], dtype=torch_dtype).numpy().dtype


class H5DataLoader(DataLoaderBase):
    """
    Read the specified dataset from rank-0,
    broadcast or distribute to other ranks.
    """

    def __init__(self,
                 h5_file_path: str, h5_dataset_path: str,
                 *,
                 # Optional reading configs for users to load the dataset.
                 dtype: Optional[torch.dtype] = None,
                 **h5_file_kwargs) -> None:
        super().__init__()

        self._unexpanded_file_path = h5_file_path
        self._dataset_path = h5_dataset_path
        self._file_kwargs = h5_file_kwargs
        self._target_dtype = dtype

    def collective_init(self, hint_name) -> None:
        check_collective_equality(
            f"The target dtype of {hint_name}", self._target_dtype
        )
        # Since H5 paths are only required on rank-0, let's not check them
        # collectively.

        dist_env = get_runtime_dist_env()
        if dist_env.rank == 0:
            self._file_path = os.path.expanduser(self._unexpanded_file_path)

            with self._dataset() as d:
                assert isinstance(d, h5py.Dataset)
                raw_np_dtype = cast(np.dtype, d.dtype)

                self.shape = tuple(d.shape)

            if self._target_dtype is not None:
                self._target_np_dtype = \
                    torch_dtype_to_numpy_dtype(self._target_dtype)
                self.dtype = self._target_dtype
            else:
                self._target_np_dtype = raw_np_dtype
                raw_dtype = numpy_dtype_to_torch_dtype(raw_np_dtype)
                self.dtype = raw_dtype

            dist_env.broadcast_object_list(0, [self.dtype, self.shape])

        else:
            [self.dtype, self.shape] = dist_env.broadcast_object_list(0)

    @contextmanager
    def _dataset(self):
        """
        Temporarily open the H5 File and Dataset.
        After reading, the dataset should be closed in time to free memeory.
        """
        dist_env = get_runtime_dist_env()
        assert dist_env.rank == 0

        with h5py.File(self._file_path, 'r', **self._file_kwargs) as f:
            d = f[self._dataset_path]
            if not isinstance(d, h5py.Dataset):
                raise TypeError()

            if self._target_np_dtype is not None:
                # NOTE the result type of `astype` has no attr `.shape/dtype`.
                d = d.astype(self._target_np_dtype)

            yield d

    def partially_load_by_chunk(self, chunk_size: int
                                ) -> Iterator[torch.Tensor]:
        cpu_dist_env = get_cpu_dist_env()
        assert cpu_dist_env.rank == 0, \
            "Loading-by-chunk is only available on rank-0"

        orig_len = self.shape[0]

        # Put tailing elements in an individual chunk whose size is smaller.
        nchunk, remainder = divmod(orig_len, chunk_size)
        if remainder > 0:
            nchunk += 1

        with self._dataset() as d:
            for i in range(nchunk):
                start = chunk_size * i
                end = min(orig_len, chunk_size * (i + 1))

                chunk_np: np.ndarray = d[start:end]
                chunk: torch.Tensor = torch.from_numpy(chunk_np)
                yield chunk

    def partially_load_by_rank(self) -> Tuple[torch.Tensor, int, int]:
        dist_env = get_cpu_dist_env()
        rank = dist_env.rank

        orig_len = self.shape[0]
        sub_shape = self.shape[1:]

        # To avoid OOM, we cannot load the whole dataset on rank-0 then
        # simply call dist.scatter.
        # Instead, we load the part for each rank once, and do P2P.
        if rank == 0:
            with self._dataset() as d:
                for w in range(1, dist_env.world_size):
                    start, end = _get_offset_exactly_nparts(
                        orig_len, nparts=dist_env.world_size, part=w)

                    part_np: np.ndarray = d[start:end]
                    part: torch.Tensor = torch.from_numpy(part_np).to(dist_env.comm_device)
                    isend = dist_env.def_isend(part, dst=w, tag=w)
                    for req in dist_env.batch_isend_irecv([isend]):
                        req.wait()

                    # TODO each rank-0-rank-w comm may take a while,
                    # subsequennt recvs should not timeout.

                s0, e0 = _get_offset_exactly_nparts(
                    orig_len, nparts=dist_env.world_size, part=0)
                part0_np: np.ndarray = d[s0:e0]
                part0 = torch.from_numpy(part0_np)
                return part0, s0, e0

        else:
            start, end = _get_offset_exactly_nparts(
                orig_len, dist_env.world_size, rank)
            buffer = torch.empty((end - start,) + sub_shape, dtype=self.dtype, device=dist_env.comm_device)
            irecv = dist_env.def_irecv(buffer, src=0, tag=rank)
            for req in dist_env.batch_isend_irecv([irecv]):
                req.wait()

            return buffer.cpu(), start, end

    def partially_load_by_index(
        self, index: torch.Tensor, *,
        chunk_size=1024 * 1024 * 128  # roughly 128M elements
    ) -> torch.Tensor:
        """
        Each time, rank-0 broadcasts a chunk [chunk_size*i, chunk_size*(i+1))
        to all ranks, and each rank picks the part it needs by
        intersecting with `index`.

        Args:
        - index: element index in the global index space, may be not ordered.
        """
        assert index.device == torch.device('cpu')

        sorted_index, sort_pos = torch.sort(index, stable=True)

        dist_env = get_cpu_dist_env()

        orig_len = self.shape[0]
        sub_shape = self.shape[1:]

        # Put tailing elements in an individual chunk whose size is smaller.
        nchunk, remainder = divmod(orig_len, chunk_size)
        if remainder > 0:
            nchunk += 1

        local_parts = []
        rev_poses = []

        def _run(d):
            for i in range(nchunk):
                start = chunk_size * i
                end = min(orig_len, chunk_size * (i + 1))

                if dist_env.rank == 0:
                    chunk_np: np.ndarray = d[start:end]
                    chunk: torch.Tensor = torch.from_numpy(chunk_np).to(dist_env.comm_device)
                    chunk = dist_env.broadcast(src=0, tensor=chunk)

                else:
                    # similar to halo calculation in dist_pass,
                    # but the chunk is defined by a pair (start, end)
                    # TODO therefore for sparse cases
                    # we can use P2P instead of broadcasting.
                    chunk = dist_env.broadcast(
                        src=0, shape=(end - start,) + sub_shape,
                        dtype=self.dtype)

                chunk = chunk.cpu()

                region = torch.logical_and(
                    sorted_index >= start, sorted_index < end)
                local_idx = sorted_index[region] - start

                local_part = chunk[local_idx]
                local_parts.append(local_part)

                rev_pos = sort_pos[region]
                rev_poses.append(rev_pos)

            if len(local_parts) == 0:
                return torch.empty((0,) + self.shape[1:], dtype=self.dtype)
            else:
                data = torch.concat(local_parts)
                pos = torch.concat(rev_poses)

                res = torch.empty_like(data)
                res[pos, ...] = data
                return res

        if dist_env.rank == 0:
            with self._dataset() as d:
                return _run(d)
        else:
            return _run(None)

    def fully_load(self, device: Union[torch.device, str]) -> torch.Tensor:
        dist_env = get_cpu_dist_env()

        if dist_env.rank == 0:
            with self._dataset() as d:
                t = torch.from_numpy(d[...]).to(dist_env.comm_device)
                dist_env.broadcast(0, t)
        else:
            t = dist_env.broadcast(0, shape=self.shape, dtype=self.dtype)

        return t.to(device=device)

    def __repr__(self) -> str:
        return ''.join([
            f'{self.__class__.__name__}(',
            # TODO we didn't escape the path strings properly
            f'h5_file_path={self._file_path}, ',
            f'h5_dataset_path={self._dataset_path}, ',
            f'dtype={self.dtype}',
            ')'
        ])


class FulledTensorLoader(DataLoaderBase):
    def __init__(self, value: Union[int, float], shape, dtype, device) -> None:
        super().__init__()

        self.value = value
        self._shape = tuple(shape)
        self._dtype = dtype
        self._device = torch.device(device)

        _check_dtype_shape_and_more(self, self.value)

    def _full(self, batch_dim_len: Optional[int] = None,
              device: Union[torch.device, str, None] = None):
        if batch_dim_len is None:
            batch_dim_len = self._shape[0]

        shape = (batch_dim_len,) + self._shape[1:]
        return torch.full(
            shape, self.value, dtype=self.dtype, device=device)  # type: ignore

    def partially_load_by_chunk(self, chunk_size: int
                                ) -> Iterator[torch.Tensor]:
        cpu_dist_env = get_cpu_dist_env()
        assert cpu_dist_env.rank == 0, \
            "Loading-by-chunk is only available on rank-0"

        orig_len = self.shape[0]

        # Put tailing elements in an individual chunk whose size is smaller.
        nchunk, remainder = divmod(orig_len, chunk_size)
        if remainder > 0:
            nchunk += 1

        for i in range(nchunk):
            start = chunk_size * i
            end = min(orig_len, chunk_size * (i + 1))

            chunk = self._full(end - start, 'cpu')
            yield chunk

    def partially_load_by_rank(self) -> Tuple[torch.Tensor, int, int]:
        cpu_dist_env = get_cpu_dist_env()
        rank = cpu_dist_env.rank
        orig_len = self._shape[0]
        start, end = _get_offset_exactly_nparts(
            orig_len, cpu_dist_env.world_size, rank)
        return self._full(end - start, 'cpu'), start, end

    def partially_load_by_index(self, index: torch.Tensor,
                                **kwargs) -> torch.Tensor:
        return self._full(index.shape[0], 'cpu')

    def fully_load(self, device: Union[torch.device, str, None]
                   ) -> torch.Tensor:
        return self._full(None, device=device or self.device)


    def __repr__(self) -> str:
        return ''.join([
            f'{self.__class__.__name__}(',
            f'value={self.value}, ',
            f'shape={self.shape}, ',
            f'dtype={self.dtype}',
            ')'
        ])


class ArangeTensorLoader(DataLoaderBase):
    def __init__(self, start: int, end: int, step: int, dtype, device) -> None:
        super().__init__()

        self._start = start
        self._end = end
        self._step = step

        length = len(range(start, end, step))
        self._shape = (length,)
        self._dtype = dtype
        self._device = torch.device(device)

        _check_dtype_shape_and_more(self, start, end, step)

    def partially_load_by_chunk(self, chunk_size: int
                                ) -> Iterator[torch.Tensor]:
        cpu_dist_env = get_cpu_dist_env()
        assert cpu_dist_env.rank == 0, \
            "Loading-by-chunk is only available on rank-0"

        orig_len = self.shape[0]

        # Put tailing elements in an individual chunk whose size is smaller.
        nchunk, remainder = divmod(orig_len, chunk_size)
        if remainder > 0:
            nchunk += 1

        for i in range(nchunk):
            range_start = self._start + chunk_size * i * self._step
            range_end = range_start + chunk_size * self._step
            if self._step > 0:
                range_end = min(self._end, range_end)
            else:
                range_end = max(self._end, range_end)

            chunk = torch.arange(range_start, range_end, self._step,
                                 dtype=self._dtype, device='cpu')
            yield chunk

    def partially_load_by_rank(self) -> Tuple[torch.Tensor, int, int]:
        cpu_dist_env = get_cpu_dist_env()
        rank = cpu_dist_env.rank
        orig_len = self._shape[0]
        offset_start, offset_end = _get_offset_exactly_nparts(
            orig_len, cpu_dist_env.world_size, rank)

        range_start = self._start + offset_start * self._step
        range_end = self._start + offset_end * self._step
        if self._step > 0:
            range_end = min(self._end, range_end)
        else:
            range_end = max(self._end, range_end)

        chunk = torch.arange(range_start, range_end, self._step,
                             dtype=self._dtype, device='cpu')
        return chunk, offset_start, offset_end

    def partially_load_by_index(self, index: torch.Tensor,
                                **kwargs) -> torch.Tensor:
        return (index * self._step + self._start).to(dtype=self._dtype)

    def fully_load(self, device: Union[torch.device, str, None]
                   ) -> torch.Tensor:
        return torch.arange(self._start, self._end, self._step,
                            dtype=self._dtype, device=device or self.device)


    def __repr__(self) -> str:
        return ''.join([
            f'{self.__class__.__name__}(',
            f'start={self._start}, ',
            f'end={self._end}, ',
            f'step={self._step}, ',
            f'dtype={self.dtype}',
            ')'
        ])
