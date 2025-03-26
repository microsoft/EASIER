# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List, Optional, Sequence
from typing_extensions import Literal, TypeAlias
import typing

import torch
from torch import LongTensor

from easier.core.runtime.dist_env import get_runtime_dist_env

from easier.core.utils import logger


class HaloExchanger(torch.nn.Module):
    """
    HaloExchanger is created and the recv buffers are preallocated for
    each FX Node of esr.Selector/Reducer.

    The resultant tensor of this Module is a _chunk_ containing all
    necessary sub-tensors for that worker-local Selector/Reducer instance,
    exactly matching the JIT-rewritten instance regarding the 
    operation semantics of `Selector/Reducer` primitive.
    """

    def __init__(self,
                 is_for_selector: bool,
                 input_elempart_length: int,  # for Selector
                 runtime_halos_lidxes: List[torch.Tensor],
                 runtime_recv_lengths: List[int],
                 parent_primitive: str
                 ) -> None:
        super().__init__()

        self.is_for_selector = is_for_selector
        self.input_elempart_length = input_elempart_length

        # Tensors in halos_lidxes are waiting to be moved to runtime device,
        # we keep the whole List so that the movement can be seen within
        # HaloExchannger.forward().
        self.runtime_halos_lidxes = runtime_halos_lidxes

        self.runtime_recv_lengths = runtime_recv_lengths

        # The attribute path of the Selector/Reducer
        # (may be a nested path, if user calls S/R in a sub-esr.Module)
        # this HaloExchanger is inserted for.
        # Would always pick the `target: str` from the S/R callsite Nodes
        # i.e. the path is always regarded to the esr.Module this
        # HaloExchanger instance will be added to.
        self.parent_primitive = parent_primitive

        self.analyze_halo_properties()

        # Callers should check this flag, only if it's needed,
        # use this HaloExchanger and append it to the IR.
        self.is_needed: bool

        # =======================
        # Fields only available if is_needed:

        # Both send and recv lengths do not count the local-to-local comm.
        self.total_recv_length: int
        self.total_send_length: int

        # None means we don't need to concat into the chunk.
        self.concat_buffer_length: Optional[int]

        # The size of the list is `world_size`.
        # If no communication is needed, the item is None.
        # `recv_buffers[this_rank]` is always None.
        self.recv_buffers: List[Optional[torch.Tensor]]
        self.concat_buffer: Optional[torch.Tensor] = None

    def analyze_halo_properties(self):
        # In certain cases we should avoid preallocation or avoid inserting
        # HaloExchanger at all:
        # - Selector/Reducer has no sends/recvs:
        #       Avoid inserting HaloExchanger
        # - Selector has no recvs:
        #       Avoid preallocation of recv_buffers and chunk

        dist_env = get_runtime_dist_env()

        self.total_recv_length = sum(
            halo_len for u, halo_len in enumerate(self.runtime_recv_lengths)
            if u != dist_env.rank
        )

        self.total_send_length = sum(
            lidx.shape[0] for t, lidx in enumerate(self.runtime_halos_lidxes)
            if t != dist_env.rank
        )

        self.is_needed = not (
            self.total_recv_length == 0 and self.total_send_length == 0
        )

        if self.is_needed:
            local_size = self.runtime_halos_lidxes[dist_env.rank].shape[0]

            if self.is_for_selector:
                if self.total_recv_length == 0:
                    # For Selector having no recvs, we don't use chunk
                    # but directly go on with the input tensor.
                    self.concat_buffer_length = None
                else:
                    # If we need to recv halos for Selector,
                    # we don't slice input tensor for local halo of Selector,
                    # but directly take the local input elempart.
                    if local_size > 0:
                        local_size = self.input_elempart_length

                    self.concat_buffer_length = \
                        self.total_recv_length + local_size
            else:
                # For Reducer, if this HaloExchanger is needed, we'll either
                # send or recv, in both cases we need concat with halos.
                self.concat_buffer_length = self.total_recv_length + local_size

    def assert_is_needed(self):
        assert self.is_needed, \
            "Selector or Reducer, that has no recvs and no sends," \
            " shouldn't have HaloExchanger at all." \
            " Avoid adding HaloExchanger in dataflow_distribution pass."

    def prepare_buffers(self, element_tensor_shape: tuple, dtype: torch.dtype):
        self.assert_is_needed()

        if self.concat_buffer_length is None:
            # For Selector having no recvs, we don't use chunk but directly
            # go on with the input tensor.
            return

        if self.concat_buffer != None:
            prev_subshape = tuple(self.concat_buffer.shape[1:])
            if prev_subshape == element_tensor_shape \
                    and self.concat_buffer.dtype == dtype:
                return

        def _get_buffer_shape(batchsize: int):
            return (batchsize,) + element_tensor_shape

        dist_env = get_runtime_dist_env()
        device = dist_env.comm_device

        self.recv_buffers: List[Optional[torch.Tensor]] = \
            [None] * dist_env.world_size
        for u in range(dist_env.world_size):
            uninum = self.runtime_recv_lengths[u]
            if u != dist_env.rank:
                if uninum > 0:
                    buf = torch.empty(
                        size=_get_buffer_shape(uninum),
                        dtype=dtype, device=device
                    )
                    self.recv_buffers[u] = buf

            # recv_buffers[dist_env.rank] is always None.

        # Because we are assigning an individual HaloExchanger per Node,
        # we can preallocate its resultant tensor memory.
        # During a single execution of the JIT-ed module,
        # this memory will be written once.
        #
        # WARNING But if the HaloExchanger instance is called twice,
        # the 2nd writing may invalidate the 1st result.
        self.concat_buffer = torch.empty(
            size=_get_buffer_shape(self.concat_buffer_length),
            dtype=dtype, device=device
        )

    def forward(self, local: torch.Tensor) -> torch.Tensor:
        self.assert_is_needed()
        self.prepare_buffers(tuple(local.shape[1:]), local.dtype)

        dist_env = get_runtime_dist_env()
        p2p_ops = []

        for u in range(dist_env.world_size):
            lidx = self.runtime_halos_lidxes[u]
            if u != dist_env.rank:
                if lidx.shape[0] > 0:
                    isend = dist_env.def_isend(local[lidx], u, tag=u)
                    p2p_ops.append(isend)

        # When Selector has no recvs,
        # don't bother writing a chunk, but directly return the input tensor.
        if self.concat_buffer is None:
            return local

        recv_buffers: List[torch.Tensor] = []
        for u in range(dist_env.world_size):
            if u != dist_env.rank:
                buf = self.recv_buffers[u]
                if buf is not None:
                    recv_buffers.append(buf)
                    irecv = dist_env.def_irecv(buf, u, tag=dist_env.rank)
                    p2p_ops.append(irecv)

            else:
                if self.runtime_halos_lidxes[dist_env.rank].shape[0] > 0:
                    if self.is_for_selector:
                        # Concat local input tensor if any its elements
                        # contribute to Selector's result
                        recv_buffers.append(local)
                    else:
                        # Slice the exact input elements contribute to Reducer
                        # result, this is required by Reducer semantics
                        recv_buffers.append(
                            local[self.runtime_halos_lidxes[dist_env.rank]]
                        )

        for req in dist_env.batch_isend_irecv(p2p_ops):
            req.wait()

        torch.concat(recv_buffers, out=self.concat_buffer)
        return self.concat_buffer


def all_gather_into_tensor(
    send_tensor: torch.Tensor
) -> torch.Tensor:
    # Being neither Parameter nor Python literals, DistEnv cannot be present
    # on FX Graph. So we need to hide it
    # with a global instance and under a function.
    dist_env = get_runtime_dist_env()
    return dist_env.all_gather_into_tensor(send_tensor, 'concat')
