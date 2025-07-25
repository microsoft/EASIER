# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from easier.core.passes.tensor_group_partition import \
    ElemPart, ElemPartArangeIdx, ElemPartReorderedArangeIdx, ElemPartSortedIdx

from easier.core.runtime.dist_env import get_runtime_dist_env

"""
In certain cases ElemParts have some good properties,
e.g. being sorted, being simply `torch.arange()` result.
Such properties can be leveraged to boost speed of sparse_encoding analyses.

`ElemPart.idx_desc` is a (light-weight) descriptor for such properties,
and must be maintained properly during transformation on ElemParts.

Currently:
-   Cases about `torch.isin` are handled by util methods here,
    as `torch.isin` essentially traverses all data, not taking advantage of
    possible sorted-ness etc.


TODO More cases can be refined to further boost sparse_encoding.

However, in contrast to a single `torch.isin` call that used to take 90% time,
those cases are scattered among sparse_encoding and summed up to less than
50% (namely, `50%*(1-90%)`, but still several hours in large-scale settings):

P.S. the N% below are taken from a certain experiment and are reference-only.

-   In `zipsort_with_order` we sort the "orderables" (~25%)
    
    P.S. However `torch.sort` tend to return quickly, depending on how much
    its input is relatively sorted.

-   In `vector_index_of` we `torch.searchsorted` (~13%)
    
    -   In the evenly partition mode, the output ElemPart to search against
        may be just ArangeIdx or ReorderedArangeIdx,
        they are consecutive and bounded.

-   `torch.unique` (~23%)

    We do `torch.unique(SR.idx)` if we only search for intersection between
    ElemPart and S/R.idx, however we can group the S/R to unique only once.

-   Besides ElemParts, the concat-ed halo `chunk_gidx_space` are used in
    `zipsort_with_order` and `vector_index_of` too.
    Since its piece are sliced from the input ElemPart, it largely inherits
    any properties the input ElemPart has.
"""


def broadcast_elempart(src: int, elempart: ElemPart) -> ElemPart:
    dist_env = get_runtime_dist_env()

    if dist_env.rank == src:
        [idx_desc, hint] = [elempart.idx_desc, elempart.hint]
        dist_env.broadcast_object_list(src, [idx_desc, hint])

        if not isinstance(idx_desc, ElemPartArangeIdx):
            dist_env.broadcast(src, elempart.idx.to(dist_env.comm_device))

        return elempart

    else:
        [idx_desc, hint] = dist_env.broadcast_object_list(src)

        if not isinstance(idx_desc, ElemPartArangeIdx):
            idx = dist_env.broadcast(
                src, shape=(elempart.lengths[src],), dtype=elempart.idx.dtype
            ).cpu()

        else:
            idx = torch.arange(idx_desc.start, idx_desc.end)

        # Only elempart.lengths is constantly replicated
        return ElemPart(idx_desc, idx, elempart.lengths, hint)


def _isin_sorted(to_find: torch.Tensor, sorted_tests: torch.Tensor):
    """
    Use torch.searchsorted() to check is-in.

    Returns the mask.
    """
    assert to_find.ndim == 1
    assert sorted_tests.ndim == 1

    if sorted_tests.shape[0] == 0:
        return torch.zeros_like(to_find, dtype=torch.bool)

    lowerbound_indexes = torch.searchsorted(sorted_tests, to_find)

    # lowerbound_index may be N=len(sorted_tests) rather than N-1,
    # if any elem of to_find is greater than the maximum test value.
    # Which would cause directly indexing to be out-of-range.
    greater_than_mask = lowerbound_indexes == sorted_tests.shape[0]

    # Rewrite to whatever valid index (len(sorted_tests) > 0),
    # and eventually mask those positions out.
    lowerbound_indexes[greater_than_mask] = 0
    lowerbound_values = sorted_tests[lowerbound_indexes]

    return torch.logical_and(lowerbound_values == to_find, ~greater_than_mask)


def isin_elempart(gidx: torch.Tensor, elempart: ElemPart) -> torch.Tensor:
    """
    Test if elements of `gidx` are in `elempart.idx`.

    Return:
    -   the mask tensor for elements of `gidx`.
    """
    if isinstance(
        elempart.idx_desc, (ElemPartArangeIdx, ElemPartReorderedArangeIdx)
    ):
        start, end = elempart.idx_desc.start, elempart.idx_desc.end
        return torch.logical_and(start <= gidx, gidx < end)

    elif isinstance(elempart.idx_desc, ElemPartSortedIdx):
        return _isin_sorted(gidx, elempart.idx)

    else:
        return torch.isin(gidx, elempart.idx)


def elempart_isin(
    elempart: ElemPart, gidx: torch.Tensor, gidx_sorted=False
) -> torch.Tensor:
    """
    Test if elements of `elempart.idx` are in `gidx`.

    Return:
    -   the mask tensor for elements of `gidx`.
    """
    def _get_arange_mask():
        assert isinstance(
            elempart.idx_desc, (ElemPartArangeIdx, ElemPartReorderedArangeIdx)
        )

        start, end = elempart.idx_desc.start, elempart.idx_desc.end
        arange_mask = torch.zeros((end - start,), dtype=torch.bool)

        # gidx contains duplicates
        gidx_mask = torch.logical_and(start <= gidx, gidx < end)
        arange_mask[gidx[gidx_mask] - start] = True

        return arange_mask

    if isinstance(elempart.idx_desc, ElemPartArangeIdx):
        arange_mask = _get_arange_mask()
        return arange_mask

    elif isinstance(elempart.idx_desc, ElemPartReorderedArangeIdx):
        arange_mask = _get_arange_mask()
        return arange_mask[elempart.idx - elempart.idx_desc.start]

    else:
        if gidx_sorted:
            return _isin_sorted(elempart.idx, gidx)
        else:
            # torch.isin() essentially traverses to check overlapping,
            # ignoring the sorted-ness. Only call torch.isin() in the most
            # general case.
            return torch.isin(elempart.idx, gidx)


def sort_elempart(elempart: ElemPart) -> ElemPart:
    if isinstance(elempart.idx_desc, (ElemPartArangeIdx, ElemPartSortedIdx)):
        return elempart

    if isinstance(elempart.idx_desc, ElemPartReorderedArangeIdx):
        start, end = elempart.idx_desc.start, elempart.idx_desc.end
        idx_desc = ElemPartArangeIdx(start, end)
        sorted_idx = torch.arange(start, end)
    else:
        idx_desc = ElemPartSortedIdx()
        sorted_idx = elempart.idx.sort()[0]

    hint = f'{elempart.hint}:sorted'
    return ElemPart(idx_desc, sorted_idx, elempart.lengths, hint)


def reorder_elempart(
    elempart: ElemPart, reordered_idx: torch.Tensor
) -> ElemPart:
    """
    `reordered_idx` must be a permutation of `elempart.idx`.
    """
    if isinstance(
        elempart.idx_desc, (ElemPartArangeIdx, ElemPartReorderedArangeIdx)
    ):
        start, end = elempart.idx_desc.start, elempart.idx_desc.end
        idx_desc = ElemPartReorderedArangeIdx(start, end)
    else:
        idx_desc = None

    hint = f'{elempart.hint}:reordered'
    return ElemPart(idx_desc, reordered_idx, elempart.lengths, hint)
