# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import dataclasses
import enum
from typing import \
    Callable, List, Sequence, Tuple, Type, TypeVar, Union, overload, Optional
from typing_extensions import TypeAlias

import torch
from torch.fx.node import Node

import easier.core.module as esr
from easier.core.passes.utils import FX, get_called_module, tree_map
from easier.core.runtime.modules import HaloExchanger
from easier.core.utils import EasierJitException


KEY__METADATA_RUNTIME = 'easier_metadata_runtimeTensorMeta'
KEY__METADATA_VIEW_SRC = 'easier_metadata_viewSrc'

#
# TensorMetadata
# (role, shape, dtype)
#


class Role(enum.Enum):
    # Does not have batch dim
    REPLICATED = 0

    # Has batch dim, if batch size == 0 the Node is not runnable
    # (but if it's HaloExchanger, even if batch size == 0 it must be run)
    DISTRIBUTED = 1


@dataclasses.dataclass(frozen=True, eq=True)
class RuntimeTensorMeta:
    """
    Runtime metadata is associated with a Tensor instance.

    In contrast to StaticNodeMeta, at runtime we may encounter with cases
    that a single PyTorch operator returns a tuple/list of Tensors
    (e.g. `maxval, maxpos = torch.max(dim=1)`).

    Therefore we need to be aware of the potential nested structure of
    runtime metadata.
    """
    role: Role
    shape: Tuple[int, ...]
    dtype: torch.dtype


StructuredTensorMeta: TypeAlias = Union[
    RuntimeTensorMeta,
    Sequence['StructuredTensorMeta']
]


def set_node_meta(node: Node, tensor_meta: StructuredTensorMeta):
    node.meta[KEY__METADATA_RUNTIME] = tensor_meta


def get_node_meta(node: Node) -> StructuredTensorMeta:
    return node.meta[KEY__METADATA_RUNTIME]


def get_runtime_metadata_from_scalar(
    val: Union[bool, int, float]
) -> RuntimeTensorMeta:
    if isinstance(val, bool):
        return RuntimeTensorMeta(Role.REPLICATED, (), torch.bool)
    elif isinstance(val, int):
        # PyTorch Python wrapper isn't aware of Python int precision,
        # so we treat ints as current minimum int32 dtype
        # so they are compatible with any torch tensor with int-kind dtype.
        return RuntimeTensorMeta(Role.REPLICATED, (), torch.int32)
    elif isinstance(val, float):
        # Same as int32, treat Python float as current minimum float32.
        return RuntimeTensorMeta(Role.REPLICATED, (), torch.float32)
    else:
        # NOTE for types that cannot explicitly appear on `Node.args`,
        # (`torch.Tensor` is one of such types), their metadata is always
        # propagated and carried by their corresponding `Node[op='get_attr']`.
        # We don't expect to see them here.
        raise EasierJitException(
            f'Scalar value {val} of type {type(val)}'
            ' cannot have associated metadata'
        )

#
# ViewSrc
#


@dataclasses.dataclass(frozen=True, eq=True)
class ViewSrc:
    """
    `ViewSrc` are Nodes that allocate the memory blocks for the tensors.

    Also, some certain multi-result operation/Node might allocate multiple
    memory blocks/tensors. Each of these memory blocks are identified with
    an extra index for the resultant item.

    For example, if one op returns/allocates two individual tensors, subsequent
    reads/writes on one tensor will not be regarded as dependent on
    reads/writes of the other tensor. Because their view_srcs are different
    in the extra index.

    To diferentiate the Tuple[Node, int|None] for one result of multi-result op
    from Tuple[ViewSrc] for the whole nested structure,
    make ViewSrc a distinct dataclass.

    TODO if the number of being nested is more than 1, the index needs to be
    of type Optional[List[int]].
    """
    node: Node
    index: Optional[int]


# TODO we might not define StructuredViewSrc in this recursive way,
# and it is also not well supported by Python type checkers.
# A more intuitive alternative is:
#       StructuredViewSrc = ViewSrc | Dict[int, ViewSrc]
#
# or even simplier: Dict[int|None, ViewSrc]
# (when deeper nested-ness is allowed, keys become List[int])
#
# - With this definition we can "collect" by simply dict.values()
# - Perhaps suitable for TensorMeta too, but we may want to reconstruct the
#   Tuple[TensorMeta,...] in log, since it reflects the nested nature of
#   runtime value tuple better.


StructuredViewSrc: TypeAlias = Union[
    ViewSrc,
    Sequence['StructuredViewSrc'],

    # NOTE we are treating scalars as ()-shape tensors, those scalars do not
    # have allocators, and set `view_src = None` for them
    None
]


def set_node_view_src(node: Node, view_src: StructuredViewSrc):
    node.meta[KEY__METADATA_VIEW_SRC] = view_src


def get_node_view_src(node: Node) -> StructuredViewSrc:
    return node.meta[KEY__METADATA_VIEW_SRC]

#
# Tensor memory address
# (Auxiliary data to calculate ViewSrc)
#


@dataclasses.dataclass(frozen=True, eq=True)
class IndexedAddr:
    """
    IndexedAddr captures the memory address of the runtime value of a Node,
    with an extra index to indicate if the Node is multi-result.

    ViewSrc is calculated from the memory address of the tensor. I.e. if two
    tensors have the same memory address, they share the same ViewSrc.
    When the current_node is the allocator of the memory, we can 1:1 map:

        IndexedAddr(addr, index:int|None) |-> ViewSrc(current_node, index)

    If current_node is not the allocator, it means the addr was allocated
    earlier, the true ViewSrc was already stored in the stackframe.
    In such cases, IndexedAddr.index can be ignored.
    """
    addr: int
    index: Optional[int]


StructuredIndexedAddr: TypeAlias = Union[
    IndexedAddr,

    Sequence['StructuredIndexedAddr'],
    None
]


#
# Metadata relevant utils
#


_T = TypeVar('_T')
_TSentinel = TypeVar('_TSentinel')
_TLeaf = TypeVar('_TLeaf')


@overload
def collect_meta(
    meta: StructuredTensorMeta,
    f: Callable[[RuntimeTensorMeta], Union[_T, _TSentinel]] = lambda x: x,
    # use other sentinel value if None is desired.
    sentinel: _TSentinel = None
) -> List[_T]:
    # Specialized for TensorMeta
    ...


@overload
def collect_meta(
    meta: object,
    f: Callable[[_TLeaf], Union[_T, _TSentinel]] = lambda x: x,
    # use other sentinel value if None is desired.
    sentinel: _TSentinel = None,
    *,
    leaf_type: Type[_TLeaf]
) -> List[_T]:
    # General cases for whatever nested Node-meta data,
    # specialized for `leaf_type`
    ...


def collect_meta(  # type: ignore
    meta,
    f=lambda x: x,
    sentinel=None,
    *,
    leaf_type=RuntimeTensorMeta
):
    ys = []

    def _collect(x):
        if isinstance(x, leaf_type):
            y = f(x)
            if y is not sentinel:
                ys.append(y)

    _ = tree_map(meta, _collect)

    return ys


def is_node_skipped(root: esr.Module, node: Node):
    """
    This method decides if the Node can be skipped:
    -   HaloExchanger is always evaluated;
    -   Inspect the Node meta, if it is DIST and zero-length, skip it.
    """
    if node.op == FX.CALL_MODULE:
        submod = get_called_module(root, node)
        if isinstance(submod, HaloExchanger):
            return False

    meta = get_node_meta(node)

    def _get_dist_bs(x: RuntimeTensorMeta):
        if x.role == Role.DISTRIBUTED:
            return x.shape[0]
        else:
            return None
    dist_sizes = set(collect_meta(meta, _get_dist_bs, sentinel=None))
    if len(dist_sizes) == 1:
        dist_size = dist_sizes.pop()
        if dist_size == 0:
            return True

    return False
