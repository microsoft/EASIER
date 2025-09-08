# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
from typing import Callable, List, Optional, Sequence, Tuple, Union, cast

import torch

from easier.core.runtime.data_loader.base import \
    DataLoaderBase, NormalizedSlice, Num
from easier.core.runtime.data_loader.utils import \
    get_overlapping_slice, get_strides, CopyingSlicer

from easier.core.runtime.dist_env import \
    get_default_dist_env
from easier.core.runtime.utils import check_collective_equality


class StridedDataLoader(DataLoaderBase):
    def __init__(
        self,
        inner: DataLoaderBase,
        index: Sequence[Union[NormalizedSlice, int]]
    ):
        super().__init__()

        if len(index) > len(inner.shape):
            raise IndexError(
                f"Too many indices for input with ndim=={len(inner.shape)}"
            )

        for i, idx in enumerate(index):
            if isinstance(idx, int):
                if not (0 <= idx < inner.shape[i]):
                    raise IndexError(
                        f"Index {idx} is out-of-range for dimension {i}"
                        f" with length {inner.shape[i]}"
                    )

            elif isinstance(idx, NormalizedSlice):
                if idx.dimlen != inner.shape[i]:
                    raise IndexError(
                        f"Index {idx} is out-of-range for dimension {i}"
                        f" with length {inner.shape[i]}"
                    )

            else:
                raise IndexError(f"Unexpected index {idx}")

        self.norm_index = index
        self.inner = inner

        self._first_slice_dim = len(index)
        self._tensor_index: List[Union[int, slice]] = []
        shape = []
        for i, idx in enumerate(index):
            if isinstance(idx, NormalizedSlice):
                shape.append(len(idx))
                self._tensor_index.append(idx.to_slice())
                self._first_slice_dim = min(self._first_slice_dim, i)
            else:
                self._tensor_index.append(idx)

        shape += list(self.inner.shape[len(index):])

        self.shape = tuple(shape)
        self.dtype = inner.dtype
        self.device = inner.device

        # Given `self.index` may contain ints, the dimensions related to those
        # ints are discarded, i.e.:
        assert len(self.shape) <= len(self.inner.shape)

    def enumerate_children_data_loaders_and_postfixes(
        self
    ) -> Tuple[Sequence['DataLoaderBase'], Sequence[str]]:
        return ([self.inner], ['.inner'])

    def collective_init(self) -> None:
        self.coll_check_dtype_shape_devicetype()

        check_collective_equality("index", self.norm_index)

    def _get_norm_slice0(self) -> NormalizedSlice:
        idx = self.norm_index[0]
        if isinstance(idx, int):
            return NormalizedSlice(self.inner.shape[0], idx, 1, 1)
        elif isinstance(idx, NormalizedSlice):
            return idx
        else:
            assert False, 'unreachable'

    # TODO because minmax/count_unique only take dim-0 index,
    # but a StridedDataLoader may have n-d indices, we cannot simply dispatch
    # to self.inner.minmax() -- but if we take a Seq[NormSlice] then we can.
    # def minmax(self, index: Seq[NormalizedSlice]) -> Tuple[Num, Num]:
    #     composed_idx = [
    #           self.tensor_index[i].compose(index[i]) for i ...
    #     ]
    #     return self.inner.minmax(composed_idx)

    def partially_load_by_range(self, index: NormalizedSlice) -> torch.Tensor:
        idx0 = self.norm_index[0]
        tidx = list(self._tensor_index)

        if isinstance(idx0, int):
            subtensor = self.inner.partially_load_by_range(
                self._get_norm_slice0()
            )
            sub_idx0 = 0

            norm_idx = cast(
                NormalizedSlice, self.norm_index[self._first_slice_dim]
            )
            composed_idx = norm_idx.compose(index)
            tidx[self._first_slice_dim] = composed_idx.to_slice()

        elif isinstance(idx0, NormalizedSlice):
            composed_idx0 = idx0.compose(index)
            subtensor = self.inner.partially_load_by_range(composed_idx0)
            sub_idx0 = Ellipsis
        else:
            assert False, 'unreachable'

        return CopyingSlicer(subtensor)[sub_idx0, *tidx[1:]]

    def partially_load_by_index(self, index: torch.Tensor) -> torch.Tensor:
        # NOTE may dispatch to different inner.load_xxx methods,
        # it's conditioned by self.index attribute, which must be collectively
        # same to ensure the structure of call stack is collectively same too.
        idx0 = self.norm_index[0]
        tidx: List[Union[int, slice, torch.Tensor]] = list(self._tensor_index)

        if isinstance(idx0, int):
            subtensor = self.inner.partially_load_by_range(
                self._get_norm_slice0()
            )
            sub_idx0 = 0

            norm_idx = cast(
                NormalizedSlice, self.norm_index[self._first_slice_dim]
            )
            composed_idx = norm_idx.start + norm_idx.step * index
            tidx[self._first_slice_dim] = composed_idx

        elif isinstance(idx0, NormalizedSlice):
            composed_idx0 = idx0.start + idx0.step * index
            subtensor = self.inner.partially_load_by_index(composed_idx0)
            sub_idx0 = Ellipsis
        else:
            assert False, 'unreachable'

        return CopyingSlicer(subtensor)[sub_idx0, *tidx[1:]]

    def fully_load(
        self, device: torch.device, replicated: bool
    ) -> torch.Tensor:
        dist_env = get_default_dist_env()
        rank = dist_env.rank
        if replicated or rank == 0:
            return CopyingSlicer(self.inner.fully_load(device, replicated))[
                *self._tensor_index
            ]
        else:
            return self.get_placeholder(device)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}' \
            f'(inner={self.inner}, index={self.norm_index})'


class CartesianProductDataLoader(DataLoaderBase):
    def __init__(
        self,
        components: Sequence[DataLoaderBase],
        # TODO to support item datatype other than scalar, we may need 'form':
        # form: Literal['flatten', 'stack'] = 'flatten'
    ):
        super().__init__()

        nd_sizes = []

        dtypes = []
        devices = []
        for i, dl in enumerate(components):
            if len(dl.shape) != 1:
                raise ValueError(f"{i}-th input ndim != 1")
            nd_sizes.append(dl.shape[0])

            dtypes.append(dl.dtype)
            devices.append(dl.device)

        if len(set(dtypes)) != 1:
            raise ValueError("Input dtypes must be the same")
        if len(set(devices)) != 1:
            raise ValueError("Input devices must be the same")

        self.shape = (math.prod(nd_sizes), len(components))
        self.dtype = dtypes[0]
        self.device = devices[0]

        self._nd_sizes = nd_sizes
        self.components = list(components)

        self._chunk_size = 128 * 1024 * 1024

    def partially_load_by_range(self, index: NormalizedSlice) -> torch.Tensor:
        idx_tensor = cast(torch.Tensor, index.to_range(torch.arange))
        return self.partially_load_by_index(idx_tensor)

    def partially_load_by_index(self, index: torch.Tensor) -> torch.Tensor:
        nd_strides: List[int] = get_strides(self._nd_sizes).tolist()

        ret = torch.empty(
            [index.shape[0], len(self.components)],
            dtype=self.dtype
        )

        # Load by chunks for each component DataLoader, in extreme cases
        # their shapes may be like (1e9,) and (2,) and need partitioning.
        nd_sizes = torch.tensor(self._nd_sizes, dtype=torch.int64)

        chunk_size = self._chunk_size
        nd_nchunks = nd_sizes // chunk_size
        have_remainder = (nd_sizes % chunk_size) > 0
        nd_nchunks = (nd_nchunks + have_remainder.to(torch.int64)).tolist()

        chunk_ids_list: List[torch.Tensor] = []
        for idl, dl in enumerate(self.components):
            chunk_ids_list.append(torch.arange(nd_nchunks[idl]))
        # (prod(nchunks), N) -- Combinations of chunk ids
        chunk_ids_combs = torch.cartesian_prod(*chunk_ids_list)

        for icomb in range(chunk_ids_combs.shape[0]):
            chunk_ids = chunk_ids_combs[icomb]

            for idl, dl in enumerate(self.components):
                if nd_nchunks[idl] == 0:
                    continue

                chunk_id = int(chunk_ids[idl])
                comp_size = self._nd_sizes[idl]

                chunk_start = chunk_id * chunk_size
                chunk_end = min(chunk_start + chunk_size, comp_size)
                chunk = dl.partially_load_by_range(NormalizedSlice(
                    comp_size, chunk_start, 1, chunk_end - chunk_start
                ))

                stride_i = nd_strides[idl]

                comp_idx = (index // stride_i) % comp_size
                if nd_nchunks[idl] == 1:
                    # avoid calculating the mask.
                    ret[:, idl] = chunk[comp_idx]
                else:
                    idx_mask = torch.logical_and(
                        chunk_start <= comp_idx, comp_idx < chunk_end
                    )
                    ret[idx_mask, idl] = chunk[comp_idx[idx_mask]]

        return ret

    def fully_load(
        self, device: torch.device, replicated: bool
    ) -> torch.Tensor:
        dist_env = get_default_dist_env()
        rank = dist_env.rank

        # nested fully_load are all collective calls
        vectors = [
            comp.fully_load(device, replicated) for comp in self.components
        ]

        if replicated or rank == 0:
            return torch.cartesian_prod(*vectors)
        else:
            return self.get_placeholder(device)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(components={repr(self.components)})'


class ConcatDataLoader(DataLoaderBase):
    def __init__(
        self,
        components: Sequence[DataLoaderBase],
        # TODO allow to concat along dim>0
    ):
        super().__init__()

        if 1 != len(
            set((d.shape[1:], d.dtype, d.device) for d in components)
        ):
            raise ValueError(
                "Inputs to concat must have the same shape[1:]/dtype/device"
            )

        self._lengths = list(d.shape[0] for d in components)
        self.components = list(components)

        self.shape = (sum(self._lengths),) + components[0].shape[1:]
        self.dtype = components[0].dtype
        self.device = components[0].device

    def _foreach_component(self, index: NormalizedSlice, fn):
        # fn: (DataLoaderBase, _SimpleIndex) -> None
        n = len(self.components)
        components = self.components

        _l = torch.cumsum(
            torch.tensor([0] + self._lengths, dtype=torch.int64),
            dim=0
        )
        starts = _l[:n]
        ends = _l[1:]

        # if index is slice-with-negative-step, foreach in reversed.
        if index.step < 0:
            components = reversed(self.components)
            starts = starts.flip(0)
            ends = ends.flip(0)

        for i, comp in enumerate(components):
            start = int(starts[i])
            end = int(ends[i])

            concat_overlap = get_overlapping_slice(
                NormalizedSlice(self.shape[0], start, 1, end - start),
                index
            )
            if len(concat_overlap) == 0:
                comp_overlap = NormalizedSlice(comp.shape[0], 0, 1, 0)
            else:
                comp_overlap = NormalizedSlice(
                    comp.shape[0],
                    concat_overlap.start - start,
                    concat_overlap.step,
                    len(concat_overlap)
                )
            fn(comp, concat_overlap, comp_overlap)

    def minmax(self, index: NormalizedSlice) -> Tuple[Num, Num]:
        amin, amax = None, None

        def _opt_cmp(a: Optional[Num], c: Num, op: Callable[[Num, Num], Num]):
            return c if a is None else op(a, c)

        def _minmax(
            comp: DataLoaderBase,
            concat_overlap: NormalizedSlice,
            comp_overlap: NormalizedSlice
        ):
            if comp_overlap.count != 0:
                nonlocal amin, amax
                comp_min, comp_max = comp.minmax(comp_overlap)
                amin = _opt_cmp(amin, comp_min, min)
                amax = _opt_cmp(amax, comp_max, max)

        self._foreach_component(index, _minmax)

        assert amin is not None
        assert amax is not None

        return amin, amax

    def partially_load_by_range(self, index: NormalizedSlice) -> torch.Tensor:
        parts = []

        def _load_comp(
            comp: DataLoaderBase,
            concat_overlap: NormalizedSlice,
            comp_overlap: NormalizedSlice
        ):
            parts.append(comp.partially_load_by_range(comp_overlap))

        self._foreach_component(index, _load_comp)

        # If region[0].step < 0, parts will be in reversed order
        return torch.concat(parts, dim=0)

    def partially_load_by_index(self, index: torch.Tensor) -> torch.Tensor:
        ret = torch.empty(
            (index.shape[0],) + self.shape[1:], dtype=self.dtype, device='cpu'
        )

        _offset = 0
        for i, comp in enumerate(self.components):
            comp_start = _offset
            comp_end = _offset + comp.shape[0]

            mask = torch.logical_and(comp_start <= index, index < comp_end)
            comp_index = index[mask] - comp_start
            comp_slice = comp.partially_load_by_index(comp_index)

            ret[mask] = comp_slice

            _offset = comp_end

        return ret

    def fully_load(self, device: torch.device, replicated) -> torch.Tensor:
        dist_env = get_default_dist_env()
        rank = dist_env.rank

        # nested fully_load are all collective calls
        parts = [
            comp.fully_load(device, replicated) for comp in self.components
        ]

        if replicated or rank == 0:
            return torch.concat(parts)
        else:
            # Discard parts even they are placeholders too, to avoid
            # materializing the concat-ed memory.
            return self.get_placeholder(device)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(components={repr(self.components)})'


class MappedDataLoaderBase(DataLoaderBase):
    """
    This is internal to DataLoader subsystem, and not exposed to users.

    P.S. Users could define an easier.Module to do the mapped operation
    and save the result to HDF5 file then use `easier.hdf5()` API.
    """

    def __init__(
        self,
        inner: DataLoaderBase,
        subshape: Tuple[int, ...],
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.shape = (inner.shape[0],) + subshape
        self.dtype = dtype or inner.dtype
        self.device = inner.device

        self.inner = inner

    def enumerate_children_data_loaders_and_postfixes(
        self
    ) -> Tuple[Sequence['DataLoaderBase'], Sequence[str]]:
        return ([self.inner], ['.inner'])

    def map(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        The result is not necessarily on CPU, derived implementations
        should check `tensor.device`.
        """
        raise NotImplementedError()

    def _map_and_check(self, tensor: torch.Tensor) -> torch.Tensor:
        mapped = self.map(tensor)
        assert mapped.shape[1:] == self.shape[1:]
        assert mapped.dtype == self.dtype
        return mapped

    def partially_load_by_range(self, index: NormalizedSlice) -> torch.Tensor:
        tensor = self.inner.partially_load_by_range(index)
        return self._map_and_check(tensor)

    def partially_load_by_index(self, index: torch.Tensor) -> torch.Tensor:
        tensor = self.inner.partially_load_by_index(index)
        return self._map_and_check(tensor)

    def fully_load(self, device: torch.device, replicated) -> torch.Tensor:
        tensor = self.inner.fully_load(device, replicated)
        if get_default_dist_env().rank == 0 or replicated:
            return self._map_and_check(tensor)
        else:
            return self.get_placeholder()

    def __repr__(self) -> str:
        # TODO better repr?
        return f'{self.__class__.__name__}' \
            f'(inner={self.inner}, shape={self.shape}, dtype={self.dtype})'


def cartesian_product(*input: DataLoaderBase) -> DataLoaderBase:
    """
    Takes N input DataLoader component, the i-th component must have its
    shape in form of `(L_i,)`, and all components must have the same
    dtype/device.

    The result shape is `(L_0*...*L_{N-1}, N)`.
    """
    # TODO different k-dims and `form in ['flatten', 'stack']`
    return CartesianProductDataLoader(input)


def concat(inputs: Sequence[DataLoaderBase]) -> DataLoaderBase:
    """
    Takes N input DataLoader component, the i-th component must have its
    shape in form of `(L_i, k)`, and all components must have the same
    dtype/device.

    The result shape is `(L_0+...+L_{N-1}, k)`.
    """
    # TODO along dim-0 only, support other dims.
    return ConcatDataLoader(inputs)
