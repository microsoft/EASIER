# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from contextlib import contextmanager
import math
import os
from typing import List, Optional, Sequence, Tuple, Union, cast, overload
import h5py

import numpy as np
import torch

from easier.core.runtime.data_loader.base import \
    DataLoaderBase, NormalizedSlice, Num
from easier.core.runtime.data_loader.utils import \
    CopyingSlicer

from easier.core.runtime.dist_env import \
    get_default_dist_env, get_runtime_dist_env
from easier.core.runtime.utils import check_collective_equality


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
        self.device = tensor.device

        # The data is always stored as CPU tensor
        self.tensor = tensor.cpu()

    def collective_init(self) -> None:
        self.coll_check_dtype_shape_devicetype()

        def _eq_tensor(v, v0):
            # torch.allclose support broadcasting, so we need to check shapes.
            return v.shape == v0.shape and torch.allclose(v, v0)
        check_collective_equality(
            f"The input tensor of {self.easier_hint_name}",
            self.tensor,
            eq=_eq_tensor
        )

    def partially_load_by_range(self, index: NormalizedSlice) -> torch.Tensor:
        return CopyingSlicer(self.tensor)[index.to_slice()]

    def partially_load_by_index(self, index: torch.Tensor) -> torch.Tensor:
        return self.tensor[index]

    def fully_load(
        self, device: torch.device, replicated: bool
    ) -> torch.Tensor:
        dist_env = get_default_dist_env()
        rank = dist_env.rank
        if replicated or rank == 0:
            return self.tensor.to(device, copy=True)
        else:
            return self.get_placeholder(device)

    def __repr__(self) -> str:
        # TODO affected by torch print options, if the print options changed,
        # may cause esr.load rejects to load a dump.
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
                 device: Union[torch.device, str],
                 # Optional reading configs for users to load the dataset.
                 dtype: Optional[torch.dtype],
                 **h5_file_kwargs) -> None:
        """
        The constructor only is collective.
        """
        super().__init__()

        self._file_path = os.path.expanduser(h5_file_path)
        self._dataset_path = h5_dataset_path
        self._file_kwargs = h5_file_kwargs

        self.device = torch.device(device)

        self.chunk_size = 1024 * 1024 * 128  # roughly 128M elements

        dist_env = get_default_dist_env()  # runtime dist env not decided yet
        if dist_env.rank == 0:
            with h5py.File(self._file_path, 'r', **self._file_kwargs) as f:
                d = f[self._dataset_path]
                if not isinstance(d, h5py.Dataset):
                    raise TypeError()

                raw_np_dtype = cast(np.dtype, d.dtype)
                self.shape = tuple(d.shape)

            if dtype is not None:
                self._target_np_dtype = torch_dtype_to_numpy_dtype(dtype)
                self.dtype = dtype
            else:
                self._target_np_dtype = None
                self.dtype = numpy_dtype_to_torch_dtype(raw_np_dtype)

            dist_env.broadcast_object_list(
                0, [self._target_np_dtype, self.dtype, self.shape]
            )

        else:
            [self._target_np_dtype, self.dtype, self.shape] = \
                dist_env.broadcast_object_list(0)

    def collective_init(self) -> None:
        # Simply to additionally check device
        self.coll_check_dtype_shape_devicetype()

        # Since H5 paths are only required on rank-0, let's not check them
        # collectively.

    @contextmanager
    def _dataset_as_dtype(self):
        """
        Temporarily open the H5 File and cast the Dataset to the target dtype.
        After reading, the dataset should be closed in time to free memeory.

        Only callable on rank-0.
        """
        # NOTE we cannot really check rank == 0 here as in some situations
        # we haven't yet initialized the DistEnv.
        # assert rank == 0

        with h5py.File(self._file_path, 'r', **self._file_kwargs) as f:
            d = f[self._dataset_path]
            assert isinstance(d, h5py.Dataset)
            if self._target_np_dtype is not None:
                # NOTE the result type of `astype` has no attr `.shape/dtype`.
                d = d.astype(self._target_np_dtype)  # type: ignore

            yield cast(h5py.Dataset, d)

    def partially_load_by_range(self, index: NormalizedSlice) -> torch.Tensor:
        dist_env = get_runtime_dist_env()
        rank = dist_env.rank

        idx_world = dist_env.gather_object_list(0, index)  # type: ignore

        def _slice_may_flip(d: h5py.Dataset, ns: NormalizedSlice):
            # h5py Dataset indexing by slice does not allow the step to be
            # negative, therefore we first flip the slice then flip the data.
            if ns.step < 0:
                rns = ns.reverse()
                part_np = d[rns.to_slice()]
                part = torch.from_numpy(part_np)
                part = part.flip(0)
            else:
                part_np = d[ns.to_slice()]
                part = torch.from_numpy(part_np)
            return part

        # To avoid OOM, we cannot load the whole dataset on rank-0 then
        # simply call dist.scatter.
        # Instead, we load the part for each rank once, and do P2P.
        if rank == 0:
            idx_world: List[NormalizedSlice]

            with self._dataset_as_dtype() as d:
                for w in range(1, dist_env.world_size):
                    part = _slice_may_flip(d, idx_world[w])
                    part = part.to(dist_env.comm_device)
                    isend = dist_env.def_isend(part, dst=w, tag=w)
                    for req in dist_env.batch_isend_irecv([isend]):
                        req.wait()

                    # TODO each rank-0-rank-w comm may take a while,
                    # subsequennt recvs should not timeout.

                part0 = _slice_may_flip(d, index)
                return part0

        else:
            shape = (index.count,) + self.shape[1:]
            buffer = torch.empty(
                shape, dtype=self.dtype, device=dist_env.comm_device
            )
            irecv = dist_env.def_irecv(buffer, src=0, tag=rank)
            for req in dist_env.batch_isend_irecv([irecv]):
                req.wait()

            return buffer.cpu()

    def partially_load_by_index(self, index: torch.Tensor) -> torch.Tensor:
        """
        Each time, rank-0 broadcasts a chunk [chunk_size*i, chunk_size*(i+1))
        to all ranks, and each rank picks the part it needs by
        intersecting with `index`.

        Args:
        - index: element index in the global index space, may be not ordered.
        """
        sorted_index, sort_pos = torch.sort(index, stable=True)

        dist_env = get_runtime_dist_env()

        orig_len = self.shape[0]
        sub_shape = self.shape[1:]

        # Put tailing elements in an individual chunk whose size is smaller.
        nchunk, remainder = divmod(orig_len, self.chunk_size)
        if remainder > 0:
            nchunk += 1

        local_parts = []
        rev_poses = []

        def _run(d):
            for i in range(nchunk):
                start = self.chunk_size * i
                end = min(orig_len, self.chunk_size * (i + 1))

                if dist_env.rank == 0:
                    chunk_np: np.ndarray = d[start:end]
                    chunk: torch.Tensor = \
                        torch.from_numpy(chunk_np).to(dist_env.comm_device)
                    chunk = dist_env.broadcast(src=0, tensor=chunk)

                else:
                    # similar to halo calculation in dist_pass,
                    # but the chunk is defined by a pair (start, end)
                    # TODO therefore for sparse cases
                    # we can use P2P instead of broadcasting.
                    chunk = dist_env.broadcast(
                        src=0, shape=(end - start,) + sub_shape,
                        dtype=self.dtype
                    )

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
            with self._dataset_as_dtype() as d:
                return _run(d)
        else:
            return _run(None)

    def fully_load(
        self, device: torch.device, replicated: bool
    ) -> torch.Tensor:
        """
        Called by backend=='none' case, only default_dist_env is available.
        """
        dist_env = get_default_dist_env()
        rank = dist_env.rank

        if replicated:
            if rank == 0:
                with self._dataset_as_dtype() as d:
                    t = torch.from_numpy(d[...]).to(dist_env.comm_device)
                    dist_env.broadcast(0, t)
            else:
                t = dist_env.broadcast(0, shape=self.shape, dtype=self.dtype)
            return t.to(device)

        else:
            if rank == 0:
                with self._dataset_as_dtype() as d:
                    t = torch.from_numpy(d[...])
                return t.to(device)

            else:
                # We cannot call .to(device) on the placeholder
                # (therefore we have to calls .to(device) many times above)
                ph = self.get_placeholder(device)
                return ph

    def __repr__(self) -> str:
        return ''.join([
            f'{self.__class__.__name__}(',
            # TODO we didn't escape the path strings properly
            f'h5_file_path={self._file_path}, ',
            f'h5_dataset_path={self._dataset_path}, ',
            f'dtype={self.dtype}',
            ')'
        ])


class FulledDataLoader(DataLoaderBase):
    def __init__(self, value: Union[int, float], shape, dtype, device) -> None:
        super().__init__()

        self.value = value
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = torch.device(device)

    def collective_init(self) -> None:
        self.coll_check_dtype_shape_devicetype()
        check_collective_equality(
            f"fill value of {self.easier_hint_name}", self.value
        )

    def minmax(self, index) -> Tuple[Num, Num]:
        return self.value, self.value

    def count_unique(self, index) -> int:
        return 1

    def _full(self, batch_dim_len: int, device: torch.device):
        shape = (batch_dim_len,) + self.shape[1:]
        return torch.full(shape, self.value, dtype=self.dtype, device=device)

    def partially_load_by_range(self, index: NormalizedSlice) -> torch.Tensor:
        return self._full(index.count, torch.device('cpu'))

    def partially_load_by_index(self, index: torch.Tensor) -> torch.Tensor:
        return self._full(index.shape[0], torch.device('cpu'))

    def fully_load(
        self, device: torch.device, replicated: bool
    ) -> torch.Tensor:
        dist_env = get_default_dist_env()
        rank = dist_env.rank
        if replicated or rank == 0:
            return self._full(self.shape[0], device)
        else:
            return self.get_placeholder(device)

    def __repr__(self) -> str:
        return ''.join([
            f'{self.__class__.__name__}(',
            f'value={self.value}, ',
            f'shape={self.shape}, ',
            f'dtype={self.dtype}',
            ')'
        ])


class ArangeDataLoader(DataLoaderBase):
    """
    The ArangeDataLoader may choose to store
    start:float/step:float/count:int instead of
    start:float/stop:float/step:float like torch.arange.

    This is because we are building a descriptive representation of the
    arange-like data distributedly, therefore even although
    `easier.arange(start, stop, step)` API, we need to calculate
    `length=(end-start)/step`.

    However, such division may lead to rounding issue,
    e.g. in Python with `easier.linspace(0, 1, 5)`, if we do the division,
    we may get an extra, intermediate exclusive end
    `end = 1/5*6 = 1.200...002`,
    which may cause the lenghth to be 7 rather than 6.
    So whenever possible, we favor direct `count` value instead of division.
    """

    def __init__(
        self,
        start: Num,
        step: Num,
        count: int,
        dtype: torch.dtype,
        device: Union[str, torch.device]
    ):
        super().__init__()

        if dtype.is_complex:
            raise NotImplementedError("range cannot be complex")
        args = [start, step]
        if dtype.is_floating_point:
            if not all(isinstance(arg, (float, int)) for arg in args):
                raise ValueError("start/step must be number")
        else:
            if not all(isinstance(arg, int) for arg in args):
                raise ValueError("start/step must be int")
        # TODO if we allow giving floats but dtype=int cases, we need to
        # ensure the datatype cast is the same as torch/numpy.

        if math.isinf(start):
            raise ValueError(f"start cannot be {start}")
        if step == 0 or math.isinf(step):
            raise ValueError(f"step must not be {step}")
        if not (isinstance(count, int) and count >= 0):
            raise ValueError(f"count must be non-negative int")

        self._start = start
        self._step = step
        self._count = count

        self.shape = (count,)
        self.dtype = dtype
        self.device = torch.device(device)

    def collective_init(self) -> None:
        self.coll_check_dtype_shape_devicetype()
        check_collective_equality(
            f"arange of {self.easier_hint_name}",
            [self._start, self._step, self._count]
        )

    def minmax(self, index: NormalizedSlice) -> Tuple[Num, Num]:
        if self.dtype.is_floating_point:
            raise ValueError("Floating-point data is not expected")

        idx1 = index.start
        idx2 = index.start + index.step * (index.count - 1)

        v1 = self._start + self._step * idx1
        v2 = self._start + self._step * idx2

        return min(v1, v2), max(v1, v2)

    def count_unique(self, index: NormalizedSlice) -> int:
        return index.count

    def partially_load_by_range(self, index: NormalizedSlice) -> torch.Tensor:
        idx_tensor = cast(torch.Tensor, index.to_range(torch.arange))
        return self.partially_load_by_index(idx_tensor)

    def partially_load_by_index(self, index: torch.Tensor) -> torch.Tensor:
        return (index * self._step + self._start).to(dtype=self.dtype)

    def fully_load(
        self, device: torch.device, replicated: bool
    ) -> torch.Tensor:
        dist_env = get_default_dist_env()
        rank = dist_env.rank
        if replicated or rank == 0:
            return (
                torch.arange(self._count) * self._step + self._start
            ).to(dtype=self.dtype, device=device)
        else:
            return self.get_placeholder(device)

    def __repr__(self) -> str:
        return ''.join([
            f'{self.__class__.__name__}(',
            f'start={self._start}, ',
            f'step={self._step}, ',
            f'count={self._count}, ',
            f'dtype={self.dtype}',
            ')'
        ])


def hdf5(
    file: str, dataset: str,
    dtype: Optional[torch.dtype] = None,
    device: Union[torch.device, str, None] = None,
    **h5_file_kwargs
):
    """
    The call to this function must be collectively.

    Create a handle to a HDF5 dataset.

    The specified dataset must be accessible from rank-0.
    """
    if device is None:
        # TODO like torch.set_default_device()
        device = 'cpu'
    return H5DataLoader(file, dataset, dtype=dtype, device=device,
                        **h5_file_kwargs)


def full(
    size: Sequence[int],
    fill_value,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[torch.device, str]] = None
):
    """
    Args:
    - dtype: Optional[torch.dtype]:
        If None, the default dtype is `torch.int64` for integer `fill_value`
        and `torch.float64` for floating-poin `fill_value`.
    - device: Optional[torch.Device]:
        If None, the default device is `"cpu"`.
    """
    if isinstance(fill_value, int):
        default_dtype = torch.int64
    elif isinstance(fill_value, float):
        default_dtype = torch.float64
    else:
        raise TypeError('fill_value must be integer or floating-point')

    if dtype is None:
        dtype = default_dtype

    if device is None:
        # TODO like torch.set_default_device()
        device = 'cpu'
    return FulledDataLoader(fill_value, size, dtype, device)


def zeros(
    size: Sequence[int],
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[torch.device, str]] = None
):
    # TODO torch.zeros/ones can have `size` be both tuple and `*size:int`.
    """
    Args:
    - dtype: Optional[torch.dtype]:
        If None, the default dtype is `torch.int64`.
    - device: Optional[torch.Device]:
        If None, the default device is `"cpu"`.
    """
    if dtype is None:
        dtype = torch.float64
    if device is None:
        # TODO like torch.set_default_device()
        device = 'cpu'
    return full(size, 0, dtype=dtype, device=device)


def ones(
    size: Sequence[int],
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[torch.device, str]] = None
):
    """
    Args:
    - dtype: Optional[torch.dtype]:
        If None, the default dtype is `torch.int64`.
    - device: Optional[torch.Device]:
        If None, the default device is `"cpu"`.
    """
    if dtype is None:
        dtype = torch.float64
    if device is None:
        # TODO like torch.set_default_device()
        device = 'cpu'
    return full(size, 1, dtype=dtype, device=device)


def _dtype_device_like(
    input: Union[DataLoaderBase, torch.Tensor],
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[torch.device, str]] = None
) -> Tuple[torch.dtype, torch.device]:
    if dtype is None:
        dtype = input.dtype

    if device is None:
        device = input.device
    device = torch.device(device)

    return dtype, device


def full_like(
    input: Union[DataLoaderBase, torch.Tensor],
    fill_value,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[torch.device, str]] = None
):
    """
    Args:
    - dtype: Optional[torch.dtype]:
        If None, the default dtype is `torch.int64` for integer `fill_value`
        and `torch.float64` for floating-poin `fill_value`.
    - device: Optional[torch.Device]:
        If None, the default device is `"cpu"`.
    """
    size = input.shape
    dtype, device = _dtype_device_like(input, dtype, device)
    return full(size, fill_value, dtype=dtype, device=device)


def zeros_like(
    input: Union[DataLoaderBase, torch.Tensor],
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[torch.device, str]] = None
):
    """
    Args:
    - dtype: Optional[torch.dtype]:
        If None, the default dtype is `torch.int64` for integer `fill_value`
        and `torch.float64` for floating-poin `fill_value`.
    - device: Optional[torch.Device]:
        If None, the default device is `"cpu"`.
    """
    dtype, device = _dtype_device_like(input, dtype, device)
    return zeros(input.shape, dtype=dtype, device=device)


def ones_like(
    input: Union[DataLoaderBase, torch.Tensor],
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[torch.device, str]] = None
):
    """
    Args:
    - dtype: Optional[torch.dtype]:
        If None, the default dtype is `torch.int64` for integer `fill_value`
        and `torch.float64` for floating-poin `fill_value`.
    - device: Optional[torch.Device]:
        If None, the default device is `"cpu"`.
    """
    dtype, device = _dtype_device_like(input, dtype, device)
    return ones(input.shape, dtype=dtype, device=device)


@overload
def arange(end, *, dtype=None, device=None): ...
@overload
def arange(start, end, step=1, *, dtype=None, device=None): ...


def arange(*args, **kwargs):
    def _end_matcher(end, *, dtype=None, device=None):
        return (0, end, 1, dtype, device)

    def _start_end_matcher(start, end, step=1, *, dtype=None, device=None):
        return (start, end, step, dtype, device)

    def _resolve():
        try:
            return _start_end_matcher(*args, **kwargs)
        except TypeError:
            pass

        try:
            return _end_matcher(*args, **kwargs)
        except TypeError:
            pass

        raise TypeError(f'Unexpected arguments {args} to easier.arange')

    start, end, step, dtype, device = _resolve()

    for arg in [start, end, step]:
        if not isinstance(arg, (int, float)):
            raise TypeError(
                'argument to easier.arange must be integer or floating-point')

    promoted_value = start + end + step
    if isinstance(promoted_value, int):
        default_dtype = torch.int64
    elif isinstance(promoted_value, float):
        default_dtype = torch.float64
    else:
        raise TypeError(
            'argument to easier.arange must be integer or floating-point')

    if dtype is None:
        dtype = default_dtype
    if device is None:
        # TODO like torch.set_default_device()
        device = 'cpu'

    # TODO for floating numbers the division may lead to unexpected rounding
    # however this might be regarded as the nature of floating numbers.
    count = math.ceil((end - start) / step)
    count = max(0, count)

    return ArangeDataLoader(start, step, count, dtype, device)


def linspace(start, stop, num, endpoint=True, dtype=None, device=None):
    # TODO numpy.linspace support array-like arguments.
    for arg in [start, stop]:
        if not isinstance(arg, (int, float)):
            raise TypeError(
                'argument to easier.linspace must be integer or floating-point'
            )
    if not (isinstance(num, int) and num >= 0):
        raise TypeError(
            'argument `num` to easier.linspace must be >= 0'
        )

    if dtype is None:
        dtype = torch.float64

    if not dtype.is_floating_point:
        raise NotImplementedError("Not supporting ints yet")

    if device is None:
        # TODO like torch.set_default_device()
        device = 'cpu'

    if num == 0 or num == 1:
        return FulledDataLoader(start, shape=[num], dtype=dtype, device=device)

    nstep = num
    if not endpoint:
        nstep += 1
    step = (stop - start) / (nstep - 1)

    if step == 0:
        return FulledDataLoader(start, shape=[num], dtype=dtype, device=device)

    if endpoint and num > 1:
        # NOTE For floating numbers the division above may cause rounding issue
        # and the specified `stop` is not bytewise included -- because the last
        # element is calculated using `start+step*(num-1)`.
        #
        # For such cases numpy directly set the y[-1] = stop, in EASIER
        # we can make it piecewise and concat.
        from easier.core.runtime.data_loader.ops import ConcatDataLoader
        ys = ArangeDataLoader(start, step, num - 1, dtype=dtype, device=device)
        y = ArangeDataLoader(stop, step, 1, dtype=dtype, device=device)
        return ConcatDataLoader([ys, y])

        # TODO numpy.linspace ensure `stop` bytewise precise by setting y[-1],
        # however, torch.linspace ensure `stop` by generating from two
        # directions: [start, start+step,...] + reverse([stop, stop-step, ...])

    else:
        return ArangeDataLoader(start, step, num, dtype=dtype, device=device)
