# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import operator
from types import EllipsisType
from typing import \
    Callable, Dict, List, Sequence, Tuple, TypeAlias, Union, cast
import numpy
import pickle

import torch
from torch.fx.node import Node

import easier as esr
from easier.core.passes.utils import FX, get_called_module, get_attr_value
from easier.core.utils import EasierJitException
from easier.core.runtime.dist_env import get_runtime_dist_env
from easier.core.runtime.modules import HaloExchanger


class JitSkipped:
    """
    We use a special runtime object `jit_skipped = JitSkipped()` to represent
    values of skipped Nodes.
    """
    pass


jit_skipped = JitSkipped()


class JitReleased:
    """
    Represent that the reference to the runtime object,
    i.e. an instance of _RuntimeValue, has been released.
    """
    pass


jit_released = JitReleased()


_RuntimeValue: TypeAlias = Union[
    torch.Tensor,
    Sequence['_RuntimeValue'],

    # NOTE it's possible that FX trace `Tensor.item()` call which results in
    # a pure int/float scalar rather than a [0]-shape tensor.
    int, float, bool,
    # plus types may appear as constants or indexes in the parameter list
    # of an operator.
    str, slice, EllipsisType, None,
]
RuntimeValue: TypeAlias = Union[
    _RuntimeValue,
    None,  # output Nodes, nested esr.Module calls
    JitSkipped,  # Skipped won't be nested
    JitReleased
]


def get_aggregator_neutral_value(aggregator, dtype: torch.dtype):
    if dtype.is_complex:
        raise NotImplementedError()

    if dtype.is_floating_point:
        finfo = torch.finfo(dtype)
        vmax = finfo.max
        vmin = finfo.min
    else:
        iinfo = torch.iinfo(dtype)
        vmax = iinfo.max
        vmin = iinfo.min

    vneutral = {
        esr.sum: 0,
        esr.prod: 1,
        esr.norm: 0,
        esr.max: vmin,
        esr.min: vmax
    }[aggregator]
    return vneutral


def exchange_meta_for_halo_exchanger(
    halo_xchg: HaloExchanger,
    input: RuntimeValue
) -> Tuple[Tuple[int, ...], torch.dtype]:
    """
    Exchange shape/dtype info for recv buffers of HaloExchangers.

    On some workers the batch size of the input ElemPart is zero and the input
    Node is skipped, then we cannot get a valid input Tensor to the halo_xchg.
    However, the halo_xchg may need to receive, then it need valid shape/dtype
    info the allocate the recv buffers.
    For such cases, we'll exchange the shape/dtype info from other ranks.

    NOTE
    -   HaloExchanger itself is not a fully collective call, it may be called
        on some ranks and not on others.
        Therefore the call to this function may not be a fully collective call.

    -   Keep using P2P with the same src/dst ranks as the halo_xchg.
        Because on the ranks without HaloExchangers, they may have entered
        and been waiting in dist.all_gather_into_tensor (the high-level API)
        or ncclAllGathr (the low-level API) etc.
        Any communication APIs than P2P may incorrectly be mixed with them.
    """
    dist_env = get_runtime_dist_env()

    # Flags reflecting halo_xchg's original P2P connectivity.
    #
    # Remarkably, if a rank has zero-batch-size input, it cannot send or
    # be received-from (but if can recv from others).
    can_send_to = torch.zeros((dist_env.world_size,), dtype=torch.bool)
    can_recv_from = torch.zeros((dist_env.world_size,), dtype=torch.bool)
    for u in range(dist_env.world_size):
        lidx = halo_xchg.runtime_halos_lidxes[u]
        if u != dist_env.rank:
            if lidx.shape[0] > 0:
                can_send_to[u] = True
    for u in range(dist_env.world_size):
        recv_len = halo_xchg.runtime_recv_lengths[u]
        if u != dist_env.rank:
            if recv_len > 0:
                can_recv_from[u] = True

    def _exchange(
        to_send: torch.Tensor,
    ):
        """
        Always exclude the self rank.
        The result will have the shape `(world_size,) + tosend.shape`.
        """
        to_send = to_send.to(dist_env.comm_device)
        recv_buffer = torch.empty(
            (dist_env.world_size,) + to_send.shape,
            dtype=to_send.dtype,
            device=dist_env.comm_device
        )
        p2p_ops = []

        for u in range(dist_env.world_size):
            if can_send_to[u]:
                isend = dist_env.def_isend(to_send, u, tag=u)
                p2p_ops.append(isend)

        for u in range(dist_env.world_size):
            if can_recv_from[u]:
                irecv = dist_env.def_irecv(
                    recv_buffer[u], u, tag=dist_env.rank
                )
                p2p_ops.append(irecv)

        for req in dist_env.batch_isend_irecv(p2p_ops):
            req.wait()
        return recv_buffer.cpu()

    #
    # Exchange dtype and ndim
    # both have relatively constant sizes
    #

    # A big enough buffer to store the serialized dtype.
    # buffer[0] is the length of bytes.
    dtype_buffer = torch.zeros((1000,), dtype=torch.int64)
    ndim = 0

    if isinstance(input, torch.Tensor):

        dtype_bytes = pickle.dumps(input.dtype)
        assert len(dtype_bytes) < dtype_buffer.shape[0] - 1
        dtype_buffer[0] = len(dtype_bytes)
        dtype_buffer[1:(1 + len(dtype_bytes))] = torch.from_numpy(
            numpy.frombuffer(dtype_bytes, dtype=numpy.uint8).copy()
        )

        ndim = input.ndim

    else:
        if not isinstance(input, JitSkipped):
            raise EasierJitException(
                f"runtime value {input} of type {type(input)} is not expected"
            )

        if not torch.any(can_recv_from):
            raise EasierJitException(
                "Unexpected HaloExchanger without any input"
            )

    dtypes_buffer = _exchange(dtype_buffer)
    ndims_buffer = _exchange(torch.tensor([ndim], dtype=torch.int64))

    # nzep stands for Non-Zero ElemPart
    nzep_dtypes = set()
    if can_recv_from.any():
        # split returns an empty tensor if input is empty
        for nzep_dtype_buffer in dtypes_buffer[can_recv_from].split(1, dim=0):
            # dtypes_buffer: (N, 1000)
            # split: [(1,1000), (1,1000), ...]
            nzep_dtype_buffer = nzep_dtype_buffer[0]
            u8_len = int(nzep_dtype_buffer[0])
            nzep_dtypes.add(pickle.loads(
                nzep_dtype_buffer[
                    1:(1 + u8_len)
                ].to(torch.uint8).numpy(force=True).tobytes()
            ))

    #
    # - Validate dtype and ndim are consistent among workers
    #   involved in the halo_xchg;
    # - Exchange shape, the size is depended on ndim
    #
    if isinstance(input, torch.Tensor):
        # Validate dtype with others, if there are any
        if not all(d == input.dtype for d in nzep_dtypes):
            raise EasierJitException(
                "dtypes of HaloExchanger are not the same:"
                f" {nzep_dtypes}"
            )
        dtype = input.dtype

        # Validate ndim with others, if there are any
        if not torch.all(ndims_buffer[can_recv_from] == ndim):
            raise EasierJitException(
                "ndim of HaloExchanger are not the same:"
                f" {ndims_buffer[can_recv_from]}"
            )

        shape_buffer = torch.tensor(input.shape, dtype=torch.int64)

    else:
        assert isinstance(input, JitSkipped)

        # Unique dtype
        if len(nzep_dtypes) != 1:
            raise EasierJitException(
                "dtypes of HaloExchanger are not the same:"
                f" {nzep_dtypes}"
            )
        dtype = nzep_dtypes.pop()

        # Unique ndim
        nzep_ndims = ndims_buffer[can_recv_from].unique()
        if nzep_ndims.shape[0] > 1:
            raise EasierJitException(
                "ndim of HaloExchanger are not the same:"
                f" {nzep_ndims}"
            )

        ndim = int(nzep_ndims[0])
        shape_buffer = torch.full((ndim,), -1, dtype=torch.int64)

    shapes_buffer = _exchange(shape_buffer)
    if isinstance(input, torch.Tensor):
        # Validate shape[1:] with others, if there are any
        if not torch.all(
            shapes_buffer[can_recv_from][:, 1:] == shape_buffer[1:]
        ):
            raise EasierJitException(
                "shape[1:] of HaloExchanger are not the same:"
                f" {shapes_buffer[can_recv_from][:, 1:]}"
            )
        subshape = tuple(input.shape[1:])

    else:
        assert isinstance(input, JitSkipped)

        # Unique shape[1:]
        nzep_subshape_buffer = \
            shapes_buffer[can_recv_from][:, 1:].unique(dim=0)
        if nzep_subshape_buffer.shape != (1, ndim - 1,):
            raise EasierJitException(
                "shape[1:] of HaloExchanger are not the same:"
                f" {nzep_subshape_buffer}"
            )
        subshape = tuple(nzep_subshape_buffer[0].tolist())

    return subshape, dtype


def allgather_meta_for_collective_input(
    input: RuntimeValue
) -> Tuple[Tuple[int, ...], torch.dtype]:
    """
    This will be a fully collective call, all ranks must be involved.

    Available scenarios include EASIER aggregators and Reducers.
    """
    dist_env = get_runtime_dist_env()

    if isinstance(input, torch.Tensor):
        arg_skipped = torch.tensor([0], device=dist_env.comm_device)

    elif isinstance(input, JitSkipped):
        arg_skipped = torch.tensor([1], device=dist_env.comm_device)

    else:
        raise EasierJitException(
            f"runtime value {input} of type {type(input)} is not expected"
        )

    # The first communication API must be fully collective, to avoid getting
    # mixed with P2P etc.
    arg_skipped_flags = dist_env.all_gather_into_tensor(arg_skipped)

    # at least one rank has shape info
    info_sender = (arg_skipped_flags == 0).argwhere().ravel()[0]

    if info_sender == dist_env.rank:
        [dtype, subshape] = dist_env.broadcast_object_list(
            info_sender,
            [input.dtype, input.shape[1:]]  # type: ignore
        )
    else:
        [dtype, subshape] = dist_env.broadcast_object_list(
            info_sender
        )

    if isinstance(input, torch.Tensor):
        if input.shape[1:] != subshape:
            raise EasierJitException(
                "shape[1:] of collective inputs are not the same:"
                f" {input.shape[1:]} and {subshape}"
            )
        if input.dtype != dtype:
            raise EasierJitException(
                "dtype of collective inputs are not the same:"
                f" {input.dtype} and {dtype}"
            )

    return tuple(subshape), dtype


def evaluate_node(
    root: esr.Module,
    node: Node,
    args: List[RuntimeValue],
    kwargs: Dict[str, RuntimeValue]
) -> RuntimeValue:
    """
    Simply evaluate the Node using runtime input values.

    All inputs must be valid tensor data. JitSkipped is not expected here.
    """
    if node.op == FX.GET_ATTR:
        res = get_attr_value(root, node)

    elif node.op == FX.CALL_FUNCTION:
        function = cast(Callable, node.target)
        res = function(*args, **kwargs)

        if function is operator.setitem:
            # By default operator.setitem will return None.
            # However torch ops have the convention that if an op modifies
            # its inputs in-place, those inputs should be included in output.
            # We return the setitem target to follow the convention
            # (and that is a requirement for passes like data_dep_analysis)
            res = args[0]

    elif node.op == FX.CALL_METHOD:
        method_name = cast(str, node.target)
        this, *other_args = args

        if not isinstance(this, torch.Tensor):
            # TODO any cases in FX that non-tensor methods are called?
            # maybe `a.split().index(3)` -- `tuple.index` is called?
            raise EasierJitException(
                "Expect a method of torch.Tensor to be called,"
                f" but method '{method_name}' of {type(this)} is called"
            )

        this_method = getattr(this, method_name)
        # `getattr` on the instance `this` already binds the method to the obj
        # so we don't pass `this` as an argument anymore.
        res = this_method(*other_args, **kwargs)

    elif node.op == FX.CALL_MODULE:
        submod = get_called_module(root, node)
        res = submod(*args, **kwargs)

    elif node.op == FX.OUTPUT:
        res = None

    else:
        assert False, f"Unexpected FX Node op {node.op}"

    return res
