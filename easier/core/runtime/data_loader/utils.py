# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from typing import Callable, List, Sequence, Type, TypeVar, cast, overload

import sympy
import torch


_T = TypeVar('_T')


@dataclass
class NormalizedSlice:
    """
    Python slice objects are relative and non-normalized descriptions,
    which are tricky to deal with in a nested manner.

    A NormalizedSlice is essentially a slice plus the length it's applied on.

    Additionally, we convert the representation from a [close,open) interval
    to start/count/step so that all fields are positive ints.

    P.S. slice objects are known to:
    -   have its `.start/.stop` be
        negative (indexed backwards on some length that's not-seen-yet)
        or None (along the direction of `.step` till the last element)
    -   good for Tensor/HDF5 indexing -- PyTorch/h5py won't treat a `slice` as
        a list of ints, but a `range` will become a list.
    """
    dimlen: int

    # strictly in [0, dimlen)
    start: int
    # positive or negative, cannot be 0
    step: int
    # always >= 0
    count: int

    def __post_init__(self):
        for k, v in self.__dict__.items():
            if not isinstance(v, int):
                raise TypeError(f"NormalizedSlice.{k} must be int")

        if not (self.dimlen >= 0):
            raise ValueError("NormalizedSlice.dimlen must be >= 0")
        if not (0 <= self.start < self.dimlen):
            raise ValueError("NormalizedSlice.start must in [0, dimlen)")
        if self.step == 0:
            raise ValueError("NormalizedSlice.step must be != 0")
        if not (self.count >= 0):
            raise ValueError("NormalizedSlice.count must be >= 0")

        if self.count > 0:
            last_idx = self.start + (self.count - 1) * self.step
            if self.step > 0:
                if not (last_idx < self.dimlen):
                    raise ValueError("NormalizedSlice is out of range")
            else:
                if not (last_idx >= 0):
                    raise ValueError("NormalizedSlice is out of range")

    def __len__(self):
        return self.count

    @staticmethod
    def from_slice(length: int, s: slice) -> 'NormalizedSlice':
        start, stop, step = s.indices(length)
        count = len(range(start, stop, step))
        return NormalizedSlice(length, start, step, count)

    def to_slice(self) -> slice:
        """
        The resultant slice is only applicable to a sequence with
        length exactly equals `self.dimlen`.
        """
        stop = self.start + self.step * self.count
        if self.step < 0 and stop < 0:
            # this mean item at 0 should be included, but the stop cannot be
            # simply (-1) -- this would mean the last item, given the semantics
            # of slice -- the true value should be (-length-1), but None is
            # equivalent given the negative step.
            stop = None
        return slice(self.start, stop, self.step)

    @overload
    def to_range(self, range_cls: Type[_T]) -> _T: ...
    @overload
    def to_range(self, range_cls: Callable) -> torch.Tensor: ...

    def to_range(self, range_cls=range):  # type: ignore
        stop = self.start + self.step * self.count
        return range_cls(self.start, stop, self.step)

    def compose(self, next: 'NormalizedSlice') -> 'NormalizedSlice':
        if not (next.dimlen == self.count):
            raise ValueError(
                'Derived DataLoader methods should maintain the alignment of'
                ' shapes during the composition of DataLoaders'
            )

        new_start = self.start + next.start * self.step
        new_step = self.step * next.step
        new_count = next.count
        return NormalizedSlice(self.dimlen, new_start, new_step, new_count)

    def split(self, chunk_size: int) -> List['NormalizedSlice']:
        nchunk, remainder = divmod(self.count, chunk_size)
        if remainder > 0:
            nchunk += 1

        splits = []
        for i in range(nchunk):
            this_size = min(self.count, chunk_size * (i + 1)) - i * chunk_size
            ns = NormalizedSlice(
                self.dimlen,
                self.start + i * chunk_size * self.step,
                self.step,
                this_size)
            splits.append(ns)
        return splits

    def reverse(self) -> 'NormalizedSlice':
        if self.count == 0:
            return self
        rstart = self.start + self.step * (self.count - 1)
        return NormalizedSlice(self.dimlen, rstart, -self.step, self.count)


def get_overlapping_slice(
    region: NormalizedSlice, selection: NormalizedSlice
) -> NormalizedSlice:
    """
    The result will have the same step sign as `selection`,
    i.e. the direction in `region` is ignored.
    """
    assert region.dimlen == selection.dimlen

    r_region = region.to_range(sympy.Range)
    r_sel = selection.to_range(sympy.Range)
    overlap = cast(sympy.Range, r_region.intersect(r_sel))

    if len(overlap) == 0:
        # overlap is a sympy.EmptySet and does not have .start/step etc. fields
        return NormalizedSlice(region.dimlen, 0, selection.step, 0)

    # sympy.Range.intersect doesn't preserve the direction/sign-of-step,
    # so we need to reverse it if s.step < 0
    if selection.step < 0:
        overlap = overlap.reversed

    return NormalizedSlice(
        region.dimlen, int(overlap.start), int(overlap.step), len(overlap)
    )


def get_strides(shape: Sequence[int]) -> torch.Tensor:
    """
    Innermost-major strides.
    P.S. use Tensor.tolist() to get a List[int] of strides.
    """
    r_shp = torch.tensor(list(shape) + [1], dtype=torch.int64).flip(dims=[0])
    r_strides = torch.cumprod(r_shp, dim=0)
    strides = r_strides[:-1].flip(dims=[0])
    return strides


class CopyingSlicer:
    """
    torch.Tensor does not support negative step or even from_numpy() on
    a numpy.ndarray with negative strides.
    Do the slicing with numpy then copy to make torch happy.

    Usage:
    ```
    CopyingSlicer(tensor)[:, ::-2, 3]
    ```
    """

    def __init__(self, tensor) -> None:
        self.tensor = tensor

    def __getitem__(self, index):
        # Rely on __getitem__ protocol and [] syntax to ease writing slices.
        if not isinstance(index, tuple):
            index = (index,)

        # one index may be torch.Tensor, can be directly used to index ndarray.
        np_v = self.tensor.numpy()[*index].copy()

        if np_v.shape == ():
            # such indexing in NumPy results in np.int64 etc. object, which
            # is esstentially a scalar, unlike in PyTorch we get 0-d tensor.
            import numpy
            np_v = numpy.array(np_v)

        return torch.from_numpy(np_v)
