# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import \
    Dict, List, Optional, Sequence, Tuple, Union, overload, TYPE_CHECKING
from typing_extensions import TypeAlias, Self
import os

# REMARK
# Because of our custom definition of `easier.sum` function,
# the builtin `sum()` is hidden.
# To use it e.g. in `EdgeTensor.dist_sync`,
# please call explicit `builtins.sum(...)` instead.
import builtins
from typing_extensions import Literal

import torch
from torch import nn
from torch import fx

import h5py

from easier.core.runtime.dist_env import \
    get_runtime_dist_env, get_default_dist_env
from easier.core.runtime.data_loader import \
    ArangeTensorLoader, DataLoaderBase, InMemoryTensorLoader, H5DataLoader, \
    FulledTensorLoader, ATTRIBUTE_PLACEHOLDER, torch_dtype_to_numpy_dtype


if TYPE_CHECKING:
    from easier.core.passes.tensor_group_partition import ElemPart
    from easier.core.passes.tensor_grouping import EasierTensorGroup


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
    return FulledTensorLoader(fill_value, size, dtype, device)


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
def arange(end, dtype=None, device=None): ...
@overload
def arange(start, end, step=1, dtype=None, device=None): ...


def arange(*args, **kwargs):
    def _end_matcher(end, dtype=None, device=None):
        return (0, end, 1, dtype, device)

    def _start_end_matcher(start, end, step=1, dtype=None, device=None):
        return (start, end, step, dtype, device)

    def _resolve():
        try:
            return _end_matcher(*args, **kwargs)
        except TypeError:
            pass

        try:
            return _start_end_matcher(*args, **kwargs)
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
    return ArangeTensorLoader(start, end, step, dtype, device)


def _resolve_data_loader(arg) -> DataLoaderBase:
    if isinstance(arg, DataLoaderBase):
        return arg
    elif isinstance(arg, Tensor) or hasattr(arg, ATTRIBUTE_PLACEHOLDER):
        # The .data is banned before it's an empty placeholder, even for cases
        # of InMemoryTensorLoader. We specifically insert a private
        # "easier_placeholder" attr on the placeholder tensors.
        raise TypeError(
            "Cannot use another easier.Tensor (or its .data property)"
            " to initialize easier.Tensor"
        )
    elif isinstance(arg, torch.Tensor):
        # This does not require it to be collective. The only collective ctors
        # e.g. H5DataLoader/esr.hdf5() are always called explicitly by users.
        return InMemoryTensorLoader(arg)
    else:
        raise TypeError(f"Unknown data type {type(arg)}")


def _dist_collect(tensor: 'Tensor') -> torch.Tensor:
    """
    Collect all distributed part of an `esr.Tensor` into its original device.

    Given that both the original device and the runtime device can be either
    CPU or GPU, and these two device choices can be different:
    Here we move tensor partitions to CPU and exchange them using CPU channels
    and then move the recovered full `torch.Tensor` to the original device.
    """
    assert tensor.elempart is not None

    dist_env = get_runtime_dist_env()
    elempart = tensor.elempart
    subshp = tuple(tensor.shape[1:])

    synced = torch.empty(
        (builtins.sum(elempart.lengths),) + subshp,  # type: ignore
        dtype=tensor.dtype, device=dist_env.comm_device
    )

    parts = dist_env.all_gather(
        tensor.data.to(dist_env.comm_device),
        shapes=[(bs,) + subshp for bs in elempart.lengths])
    idxes = dist_env.all_gather(
        elempart.idx.to(dist_env.comm_device),
        shapes=[(bs,) for bs in elempart.lengths]
    )

    for part, idx in zip(parts, idxes):
        synced[idx] = part

    return synced.to(device=tensor.easier_data_loader.device)


def _resolve_to_args(
    cls_name: Literal['Module', 'Selector', 'Reducer', 'Tensor'],
    args: tuple,
    kwargs: dict
) -> Tuple[Optional[torch.device], Optional[torch.dtype]]:
    """
    PyTorch nn.Module.to() or torch.Tensor.to() have multiple overloadings:
    - to(device)
    - to(dtype)
    - to(device, dtype)
    These overloadings are differentiated by parameter types rather than
    the number of parameters.

    TODO for the sake of simplicity, we take only one argument for now.
    To change both dtype and device, users could use a chain of calls
    `mod.to(dtype).to(device)` for now.
    """
    # TODO validate kwargs.values() are well typed, e.g. avoid device=int64
    _args = set(args).union(kwargs.values())
    if len(_args) != 1:
        raise NotImplementedError(
            f'Currently easier.{cls_name}.to() only supports one argument,'
            ' consider using method chaining `.to(dtype).to(device)`'
        )
    arg, = _args

    if isinstance(arg, torch.dtype):
        return None, arg
    elif isinstance(arg, (str, torch.device)):
        return torch.device(arg), None
    else:
        raise TypeError(
            f"easier.{cls_name}.to() does not accept the argument"
            f" of type {type(arg)}"
        )


IdxStatus: TypeAlias = Literal['placeholder', 'partially_loaded', 'rewritten']


class Selector(nn.Module):
    def __init__(self, idx: Union[torch.Tensor, DataLoaderBase]):
        super().__init__()

        self.easier_data_loader = _resolve_data_loader(idx)
        self.easier_index_status: IdxStatus = 'placeholder'

        self.idx: torch.Tensor = self.easier_data_loader.get_placeholder()

        # ======
        # Fields filled during JIT compilation
        self.easier_hint_name: str
        self.easier_tensor_group: 'EasierTensorGroup'

        # Which part of the data is initially loaded for dist_pass rewriting,
        # only needed in dist_pass.
        self.easier_idx_part_range: Optional[Tuple[int, int]] = None

        # `self.idx` is required to be registerd as a "buffer" so it's moveable
        # in non-JIT mode.
        # In contrast, halo indexes are only moved and moved once to
        # backend device during JIT.
        self.runtime_halos_local_idxes: List[torch.Tensor]
        self.runtime_halos_recv_lengths: List[int]

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.easier_index_status != 'rewritten':
            raise RuntimeError("Selector.idx not ready, run compile() first")

        # NOTE at backend==none runtime,
        # bad cases like `not(self.idx_max < tensor.shape[0])` will be
        # thrown and reported by the tensor indexing operation.
        return tensor[self.idx]

    def to(self, device: Union[str, torch.device]) -> Self:
        # easier.Module/Selector/Reducer cannot take dtype. See Reducer.to()
        if self.easier_index_status != 'placeholder':
            raise RuntimeError(
                "The properties of easier.Selector can only be modified"
                " before easier.compile()"
            )

        self.easier_data_loader = self.easier_data_loader.to(device=device)
        self.idx = self.easier_data_loader.get_placeholder()
        return self


ReduceOp: TypeAlias = Literal['sum', 'prod', 'mean', 'amax', 'amin']


class Reducer(nn.Module):

    def __init__(self, idx: Union[torch.Tensor, DataLoaderBase],
                 n: int,
                 reduce: ReduceOp = 'sum'
                 ) -> None:
        super().__init__()

        self.n = n
        self.reduce = reduce

        self.easier_data_loader = _resolve_data_loader(idx)
        self.easier_index_status: IdxStatus = 'placeholder'

        self.idx: torch.Tensor = self.easier_data_loader.get_placeholder()

        # ======
        # Fields filled during JIT compilation
        self.easier_hint_name: str
        self.easier_tensor_group: 'EasierTensorGroup'

        # Which part of the data is initially loaded for dist_pass rewriting,
        # only needed in dist_pass.
        self.easier_idx_part_range: Optional[Tuple[int, int]] = None

        self.easier_reordering_selector_idx: Optional[torch.Tensor] = None

        # `self.idx` is required to be registerd as a "buffer" so it's moveable
        # in non-JIT mode.
        # In contrast, halo indexes are only moved and moved once to
        # backend device during JIT.
        self.runtime_halos_local_idxes: List[torch.Tensor]
        self.runtime_halos_recv_lengths: List[int]

    def set_is_full(self):
        """
        # TODO move this to codegen pass
        Calculate and locally set the flag for if the local Reducer is full.

        Remark:
        The is_full flag can only be calculated after distribution rewriting
        or after fully index loading.
        """
        assert self.easier_index_status == 'rewritten', \
            "Reducer.idx must have been rewritten or loaded"

        unique_idx = self.idx.unique(sorted=True)

        self.easier_is_full: bool = \
            unique_idx.shape[0] == self.n and \
            unique_idx[0] == 0 and unique_idx[-1] == (self.n - 1)

    def forward(self, tensor: torch.Tensor,
                *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.easier_index_status != 'rewritten':
            raise RuntimeError("Reducer.idx not ready, run compile() first")

        shape = tensor.shape

        if out is None:
            out = torch.zeros(
                (self.n,) + shape[1:],
                dtype=tensor.dtype, device=tensor.device)

        if out.shape[0] != self.n:
            raise ValueError(
                f"{out.shape[0]} the length of the first dimension of `out`"
                f" and {self.n} the specified `Reducer.n` do not match")

        idx = self.idx[
            (...,) + (None,) * (len(shape) - 1)].expand(-1, *shape[1:])
        return out.scatter_reduce_(0, idx, tensor, self.reduce,
                                   include_self=False)

    def to(self, device: Union[str, torch.device]) -> Self:
        # easier.Module/Selector/Reducer cannot take int dtype.
        # While Selector (essentially doing a[b]) can have int32/int16
        # as idx dtype, Reducer (torch.scatter_reduce_) can only take int64,
        # otherwise 'torch' JIT backend won't work.
        # TODO we'd better either enforce int64 in ctors or do cast accordingly
        if self.easier_index_status != 'placeholder':
            raise RuntimeError(
                "The properties of easier.Reducer can only be modified"
                " before easier.compile()"
            )

        self.easier_data_loader = self.easier_data_loader.to(device=device)
        self.idx = self.easier_data_loader.get_placeholder()
        return self


class Tensor(nn.Parameter):

    def __new__(cls,
                data: Union[torch.Tensor, DataLoaderBase],
                mode: Literal['partition', 'replicate'],
                requires_grad: bool = False) -> "Tensor":
        dl = _resolve_data_loader(data)
        data_ph: torch.Tensor = dl.get_placeholder()
        tensor = super().__new__(cls, data_ph, requires_grad)  # type: ignore

        # store the parsing results to
        # avoid parsing args/kwargs in __init__ again.
        tensor.easier_data_loader = dl
        return tensor

    def __init__(self,
                 data: Union[torch.Tensor, DataLoaderBase],
                 mode: Literal['partition', 'replicate'],
                 requires_grad: bool = False) -> None:
        # ======
        # Fields filled during __new__

        self.easier_data_loader: DataLoaderBase

        # The value for this field is collectively same:
        # - False: esr.compile() not run for any modules containing this tensor
        # - True: data loaded and non-empty on all ranks, including:
        #    -- dist tensors in backend='torch/cpu/cuda',
        #       and these are cases where `elempart != None`
        #    -- replicated tensors in all backends.
        # - rank0_only: generally, dist tensors in backend='none' have this,
        #       meaning that all data is on rank-0, other ranks have empty.
        self.easier_data_ready: Literal[False, True, 'rank0_only'] = False

        # TODO names like `is_partition` are potentially conflict to pytorch,
        # will `easier_is_partition` with namespace be better?
        self.is_partition = mode == 'partition'
        self.is_replica = mode == 'replicate'

        # ======
        # Fields filled during JIT compilation
        self.easier_hint_name: str

        # Only distributed tensors have tensor groups
        # (no matter if they are referenced by `get_attr` Nodes),
        self.easier_tensor_group: 'EasierTensorGroup'

        # Only tensors that are distributed and in `torch` etc. backends
        # have this field set.
        self.elempart: 'Optional[ElemPart]' = None

    def __repr__(self) -> str:
        if self.is_replica:
            if not self.easier_data_ready:
                repr_str = ' ' + repr(self.easier_data_loader)
            else:
                repr_str = '\n' + repr(self.data)
            return 'Replicated easier.Tensor' + repr_str
        else:
            if not self.easier_data_ready:
                repr_str = ' ' + repr(self.easier_data_loader)
            else:
                repr_str = ''
            return 'Managed partitioned easier.Tensor' + repr_str

    def __setitem__(self, indices, val) -> Self:
        """
        This __setitem__ generally runs for setting replicated easier.Tensor
        content, outside the scope for compiled easier.Module.forward().
        """
        if not self.easier_data_ready:
            raise RuntimeError("Tensor data is not ready, run compile() first")
        self.data.__setitem__(indices, val)
        return self

    @overload
    def to(self, dtype: torch.dtype) -> Self: ...
    @overload
    def to(self, device: Union[str, torch.device]) -> Self: ...
    # TODO support this overload like torch.nn.Module.to()
    # @overload
    # def to(self, device: Union[str, torch.device], dtype: torch.dtype): ...

    def to(self, *args, **kwargs) -> Self:
        """
        Before easier.compile(), users can call easier.Tensor.to() to modify:
        -   the target dtype,
        -   the target device.

        And unlike torch.Tensor.to(), easier.Tensor.to() always return the
        easier.Tensor instance itself
        (since before easier.compile(), an easier.Tensor is only a descriptor
        of the distributed data).
        """
        if self.easier_data_ready != False:
            raise RuntimeError(
                "The properties of easier.Tensor can only be modified"
                " before easier.compile()"
            )

        device, dtype = _resolve_to_args('Tensor', args, kwargs)

        self.easier_data_loader = self.easier_data_loader.to(
            device=device, dtype=dtype
        )
        self.data = self.easier_data_loader.get_placeholder()
        return self

    def collect(self) -> torch.Tensor:
        """
        `easier.Tensor.collect()` collectively returns the full tensor
        on all ranks.

        The resultant torch.Tensor will be on the device originally specified
        by the user program before entering `easier.compile()`.
        """
        if not self.easier_data_ready:
            raise RuntimeError("Tensor data is not ready, run compile() first")

        elif self.easier_data_ready == 'rank0_only':
            dist_env = get_default_dist_env()
            if dist_env.rank == 0:
                t = dist_env.broadcast(0, self.data.to(dist_env.comm_device))
            else:
                t = dist_env.broadcast(0, shape=self.shape, dtype=self.dtype)
            return t.to(self.easier_data_loader.device, copy=True)

        else:
            assert self.easier_data_ready == True
            if self.elempart is not None:
                return _dist_collect(self)
            else:
                return self.data.to(self.easier_data_loader.device, copy=True)

    def save(self, h5_file_path, h5_dataset_path, **h5_file_kwargs) -> None:
        if not self.easier_data_ready:
            raise RuntimeError("Tensor data is not ready, run compile() first")

        # with 'none' backend we don't have runtime_dist_env configured.
        rank = get_default_dist_env().rank
        if rank == 0:

            h5_file_path = os.path.expanduser(h5_file_path)

            os.makedirs(os.path.dirname(h5_file_path), exist_ok=True)

            np_dtype = torch_dtype_to_numpy_dtype(self.dtype)
            orig_shape = self.easier_data_loader.shape

            # mode 'a', append: read/write if exists, create otherwise
            with h5py.File(h5_file_path, mode='a', **h5_file_kwargs) as h5f:
                # TODO this dataset is kept only, will it need to be
                # close and reopen (created only once) to avoid OOM?
                h5d = h5f.create_dataset(
                    h5_dataset_path, dtype=np_dtype, shape=orig_shape)

                if self.elempart is not None:
                    _dist_save(self, h5d)  # collectively save dist tensor
                else:
                    h5d[...] = self.data.cpu()  # replica or 'none' backend

        else:
            if self.elempart is not None:
                _dist_save(self, None)  # collectively save dist tensor


def _dist_save(
    tensor: 'Tensor',
    h5d: Optional[h5py.Dataset],
    *,
    chunk_size: Optional[int] = None
) -> None:
    """
    Save all distributed part of an `esr.Tensor` into a HDF5 dataset.

    The paths to the HDF5 dataset must be accessible from rank-0 process.

    On rank-0, each time it handles the range [chunk_size*i, chunk_size*(i+1))
    (chunk_size determines the maximum memory consumption on rank-0)
    and every element of this range is gathered from all ranks.
    Each time, all ranks only send the element in that range to rank-0.
    """
    assert tensor.elempart is not None

    dist_env = get_runtime_dist_env()
    chunk_size = chunk_size or 1024 * 1024 * 128  # roughly 128M elements

    orig_len = tensor.easier_data_loader.shape[0]
    sub_shape = tensor.easier_data_loader.shape[1:]

    nchunk, remainder = divmod(orig_len, chunk_size)
    if remainder > 0:
        nchunk += 1

    idx = tensor.elempart.idx

    for i in range(nchunk):
        idx_start = chunk_size * i
        idx_end = builtins.min(orig_len, chunk_size * (i + 1))

        # TODO O(orig_len * nchunk) complexity, maybe sorting elempart.idx
        # first could make it faster.
        idx_region = torch.logical_and(idx >= idx_start, idx < idx_end)
        chunk_idx = idx[idx_region].to(dist_env.comm_device) - idx_start
        chunk_data = tensor[idx_region].to(dist_env.comm_device)

        idx_chunks = dist_env.gather(0, chunk_idx)
        data_chunks = dist_env.gather(0, chunk_data)

        if dist_env.rank == 0:
            assert isinstance(idx_chunks, list)
            assert isinstance(data_chunks, list)
            assert len(idx_chunks) == len(data_chunks)

            chuck_size_i = idx_end - idx_start
            assert builtins.sum(s.shape[0] for s in idx_chunks) == chuck_size_i

            h5d_chunk = torch.empty(
                (chuck_size_i,) + sub_shape,
                dtype=tensor.dtype,
                device=dist_env.comm_device
            )
            for chunk_idx, chunk_data in zip(idx_chunks, data_chunks):
                assert chunk_idx.shape[0] == chunk_data.shape[0]
                h5d_chunk[chunk_idx] = chunk_data

            assert h5d is not None
            h5d[idx_start:idx_end] = h5d_chunk.cpu()


class Module(nn.Module):

    def __init__(self):
        super().__init__()

        # Already pruned for pickle
        self.easier_raw_graph: Optional[fx.graph.Graph] = None

        # ======
        # Fields filled during JIT compilation
        self.easier_hint_name: str

        self.easier_jit_backend: Literal[
            'torch', 'cpu', 'cuda', 'none', None
        ] = None

        # Only has value when jit_backend in ['torch','cpu','cuda']
        self.partition_mode: Literal['metis', 'evenly']

        # Each Module shares the ElemPart dict of all Modules in JIT session.
        self.easier_elemparts: 'Dict[EasierTensorGroup, ElemPart]'

    def forward(self):
        pass

    @overload
    def to(self, dtype: torch.dtype) -> Self: ...
    @overload
    def to(self, device: Union[str, torch.device]) -> Self: ...
    # TODO support this overload like torch.nn.Module.to()
    # @overload
    # def to(self, device: Union[str, torch.device], dtype: torch.dtype): ...

    def to(self, *args, **kwargs) -> Self:
        """
        Before easier.compile(), users can call easier.Module.to() to modify:
        -   The target dtype of all nested easier.Tensors.

            The input dtype must be floating points, and
            the casting only take effect on easier.Tensors that are defined to
            be floating points.

        -   The target device of all nested
            easier.Modules/Selectors/Reducers/Tensors.

        easier.Module/Selector/Reducer/Tensor.to() will be called on those
        nested objects.
        Particularly for easier.Tensors, .to() calls won't clone the
        easier.Tensors but merely modify their properties.
        This is unlike torch.Tensor.to().
        """
        if self.easier_jit_backend is not None:
            raise RuntimeError(
                "The properties of easier.Module can only be modified"
                " before easier.compile()"
            )

        device, fp_dtype = _resolve_to_args('Module', args, kwargs)
        if fp_dtype is not None:
            if not fp_dtype.is_floating_point:
                # TODO check dtype.is_complex
                raise ValueError(
                    f"easier.Module.to() does not accept the dtype {fp_dtype},"
                    " floating point dtypes are required"
                )

        # avoid circular import
        from easier.core.passes.utils import get_easier_objects

        # Keys are Modules/Selectors/Reducers/Tensors/DataLoaders,
        # remarkably:
        # - attribute DataLoaders of Modules are casted and setattr().
        # - Selectors/Reducers that not involved in forward();
        # - many S/R/Ts may share a DataLoader instance at the beginning,
        #   after .to() each of them will have an individual DataLoader,
        #   this is OK as DataLoaders are purely descriptive and lightweight,
        #   and won't affect the collective behavior of DataLoaders.
        objs: dict = get_easier_objects([self])
        for obj in objs:
            if isinstance(obj, (Selector, Reducer)):
                if device is not None:
                    obj.to(device)

            if isinstance(obj, Tensor):
                # TODO always `obj.to()` twice as Tensor.to() does not have
                # good overloading resolution.
                if device is not None:
                    obj.to(device)
                if fp_dtype is not None:
                    if obj.easier_data_loader.dtype.is_floating_point:
                        obj.to(fp_dtype)

            if isinstance(obj, Module):
                # Including self Module.
                #
                # Including DataLoaders that are purely bound on Modules
                # but never used by S/R/T (and those DTs won't be covered by
                # get_easier_objects), those DTs are casted only for
                # easier usage for users, since users somehow stored the DTs.
                #
                # TODO we do not examine DTs in user-defined collections
                # like `self.dts = (dt1, dt2)`, this behavior seems the same as
                # users should use register_buffer() or ParameterList in torch.
                for dt_name, dt in list(obj.__dict__.items()):
                    if isinstance(dt, DataLoaderBase):
                        if dt.dtype.is_floating_point:
                            dt_fp_dtype = fp_dtype
                        else:
                            dt_fp_dtype = None
                        dt = dt.to(dtype=dt_fp_dtype, device=device)
                        setattr(obj, dt_name, dt)

        return self


def _allreduce(op, tensor: torch.Tensor, *args, **kwargs):
    return op(tensor, *args, dim=0, keepdim=True, **kwargs)


def sum(tensor: torch.Tensor) -> torch.Tensor:
    """
    Equivalent to `torch.sum(vertex_tensor, dim=0, keepdim=True)`.

    Args:
    -   tensor:
        At JIT-time, must be a distributed tensor.
    """
    return _allreduce(torch.sum, tensor)


def prod(tensor: torch.Tensor) -> torch.Tensor:
    """
    Equivalent to `torch.prod(vertex_tensor, dim=0, keepdim=True)`.

    Args:
    -   tensor:
        At JIT-time, must be a distributed tensor.
    """
    return _allreduce(torch.prod, tensor)


def norm(tensor: torch.Tensor, p: Union[int, str] = 2) -> torch.Tensor:
    """
    Equivalent to `torch.norm(tensor, p, dim=0, keepdim=True)`.

    Args:
    -   tensor:
        At JIT-time, must be a distributed tensor.
    """
    return _allreduce(torch.norm, tensor, p)


def max(tensor: torch.Tensor) -> torch.Tensor:
    """
    Equivalent to `torch.amax(tensor, dim=0, keepdim=True)`.

    Args:
    -   tensor:
        At JIT-time, must be a distributed tensor.
    """
    return _allreduce(torch.amax, tensor)


def min(tensor: torch.Tensor) -> torch.Tensor:
    """
    Equivalent to `torch.amin(tensor, dim=0, keepdim=True)`.

    Args:
    -   tensor:
        At JIT-time, must be a distributed tensor.
    """
    return _allreduce(torch.amin, tensor)


easier_aggregators = (sum, prod, norm, max, min)
