# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy
from typing_extensions import Literal, TypeAlias, TypeVar
import typing
import threading
import time
import functools
import copy
import pickle

import torch
import torch.distributed as dist

from easier.core.utils import logger, EasierJitException

_T = TypeVar("_T")


def _wrap_commapi_pre_filter(prefilter, api):
    """
    Both `api` and `pre_filter` function objects are not 
    bound to some DistEnv instance yet, the `self` argument
    will be included at the head in `args` in the wrapper.
    """
    @functools.wraps(api)
    def wrapper(*args, **kwargs):
        args, kwargs = prefilter(*args, **kwargs)
        return api(*args, **kwargs)
    return wrapper


class DistEnv:
    def __init__(
        self,
        world_size: int, rank: int, local_rank: int,
        device: Union[str, torch.device]
    ) -> None:
        """
        Args:
        -   device: To serve as the channel for communication.
                Aligning with the target accelerator hardware can achieve
                minimum data movement overheads.
        """
        self.world_size = world_size
        self.rank = rank
        self.local_rank = local_rank

        # existing `torch.device` instance remains unchanged.
        self.comm_device = torch.device(device)

    def __init_subclass__(cls) -> None:
        """
        Before a communication API provided by subclass DistEnv runs,
        a _prefilter_ defined in base DistEnv will run first to check and
        transform the given arguments, to do some common precondition checks.
        """
        for member_name, member in list(cls.__dict__.items()):
            # cls.__dict__ doesn't contain inherited methods from DistEnv
            if callable(member):
                # both `member` and `pre_filter` function objects are not
                # bound to some DistEnv instance yet, the `self` argument
                # will be included at the head in `args` in the wrapper.
                pre_filter = getattr(DistEnv, '_pre_' + member_name, None)
                if pre_filter is not None:
                    setattr(
                        cls, member_name,
                        _wrap_commapi_pre_filter(pre_filter, member)
                    )

    @typing.overload
    def broadcast(self, src: int, tensor: torch.Tensor) -> torch.Tensor: ...

    @typing.overload
    def broadcast(self, src: int, *,
                  shape: Sequence[int],
                  dtype: torch.dtype) -> torch.Tensor: ...

    def broadcast(self, src: int, tensor: Optional[torch.Tensor] = None,
                  *,
                  shape: Optional[Sequence[int]] = None,
                  dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        raise NotImplementedError()

    def _pre_broadcast(self, src, tensor=None, *, shape=None, dtype=None):
        assert 0 <= src < self.world_size

        if src == self.rank:
            assert tensor is not None, \
                "Broadcast sender must provide the tensor"
            assert shape is None and dtype is None, \
                "Broadcast sender should not specify the shape or dtype"

            if not tensor.is_contiguous():
                logger.debug('Broadcasting a non-contiguous tensor')
                tensor = tensor.contiguous()

        else:
            assert tensor is None, \
                "Broadcast receiver should not provide the tensor"
            assert shape is not None and dtype is not None, \
                "Broadcast receiver must specify the shape and dtype"

        return (self, src, tensor), {'shape': shape, 'dtype': dtype}

    @typing.overload
    def broadcast_object_list(self, src: int, object_list: list) -> list: ...
    @typing.overload
    def broadcast_object_list(self, src: int) -> list: ...

    def broadcast_object_list(self, src: int,
                              object_list: Optional[list] = None
                              ) -> list:
        """
        Remark:
        When a CUDA tensor is a part of the broadcasted object, torch.dist
        will try to reconstruct it on the destination worker using
        literally the same Tensor.device, such as 'cuda:3'. Such a torch.device
        may not exist on that worker.

        To eliminate such risks, when broadcasting tensors as _objects_,
        ensure those tensors are with `device='cpu'`.
        """
        raise NotImplementedError()

    def _pre_broadcast_object_list(self, src, object_list=None):
        assert 0 <= src < self.world_size

        if src == self.rank:
            assert object_list is not None, \
                "Broadcast sender must provide the object list"
        else:
            object_list = None  # unused
        return (self, src, object_list), {}

    def def_isend(self, tensor: torch.Tensor, dst: int, tag: int) -> Any:
        """
        To achieve compatibility on different communication backends like
        GLOO and NCCL, we shouldn't directly invoke a communication call,
        but to define its description first then `DistEnv.batch_isend_irecv`.
        """
        raise NotImplementedError()

    def _pre_def_isend(self, tensor, dst, tag):
        assert 0 <= dst < self.world_size

        if not tensor.is_contiguous():
            logger.debug('Defining isend on a non-contiguous tensor')
            tensor = tensor.contiguous()
        return (self, tensor, dst, tag), {}

    def def_irecv(self, buffer: torch.Tensor, src: int, tag: int) -> Any:
        """
        To achieve compatibility on different communication backends like
        GLOO and NCCL, we shouldn't directly invoke a communication call,
        but to define its description first then `DistEnv.batch_isend_irecv`.
        """
        raise NotImplementedError()

    def _pre_def_irecv(self, buffer, src, tag):
        assert 0 <= src < self.world_size

        if not buffer.is_contiguous():
            # If the `buffer` parameter is not contiguous,
            # there is no last chance to recover and reflect to callers.
            raise EasierJitException(
                "Defining irecv on a non-contiguous tensor")
        return (self, buffer, src, tag), {}

    def batch_isend_irecv(self, p2p_ops: List[Any]) -> List[Any]:
        raise NotImplementedError()

    def send(self, tensor: torch.Tensor, dst: int, tag: int) -> None:
        """
        Blockingly send one tensor.
        """
        isend = self.def_isend(tensor, dst, tag)
        for req in self.batch_isend_irecv([isend]):
            req.wait()

    def recv(self, buffer: torch.Tensor, src: int, tag: int) -> torch.Tensor:
        """
        Blockingly receive one tensor.
        """
        irecv = self.def_irecv(buffer, src, tag)
        for req in self.batch_isend_irecv([irecv]):
            req.wait()
        return buffer

    def send_int64(self, x: int, dst: int, tag: Optional[int] = None) -> None:
        self.send(
            torch.tensor([x], dtype=torch.int64, device=self.comm_device),
            dst=dst,
            tag=tag or dst  # default tag: destination
        )

    def recv_int64(self, src: int, tag: Optional[int] = None) -> int:
        t = torch.empty((1,), dtype=torch.int64, device=self.comm_device)
        self.recv(t, src=src, tag=tag or self.rank)  # default tag: destination
        return int(t)

    def all_to_all_single(self, local_input: torch.Tensor) -> torch.Tensor:
        """
        AllToAll `local_input[rank]`.

        Currently `local_input` must have ndim==1.
        """
        raise NotImplementedError()

    def _pre_all_to_all_single(self, local_input):
        if not local_input.is_contiguous():
            logger.debug('Doing all_to_all_single on a non-contiguous tensor')
            local_input = local_input.contiguous()
        return (self, local_input,), {}

    def all_to_all(self, tensors: Sequence[torch.Tensor]
                   ) -> List[torch.Tensor]:
        """
        AllToAll a list of input tensors, specially optimized for cases where
        some input tensors are empty.

        Currently each input tensor must have ndim==1.
        """
        raise NotImplementedError()

    def _pre_all_to_all(self, tensors):
        assert len(tensors) == self.world_size
        for i, tensor in enumerate(tensors):
            assert tensor.ndim == 1
        dtypes = set(t.dtype for t in tensors)
        assert len(dtypes) == 1

        tensors = list(tensors)  # copy the list
        for i, tensor in enumerate(tensors):
            if not tensor.is_contiguous():
                # Ensure contiguousness for the last time.
                logger.debug(
                    f'The {i}-th tensor to all_to_all is non-contiguous')
                tensors[i] = tensor.contiguous()

        return (self, tensors,), {}

    def all_gather_into_tensor(self, send_tensor: torch.Tensor,
                               form: Literal['concat', 'stack'] = 'concat'):
        raise NotImplementedError()

    def _pre_all_gather_into_tensor(self, send_tensor, form='concat'):
        if not send_tensor.is_contiguous():
            # Ensure contiguousness for the last time.
            logger.debug(
                'Doing all_gather_into_tensor on a non-contiguous tensor')
            send_tensor = send_tensor.contiguous()

        return (self, send_tensor, form), {}

    def all_gather(self, send_tensor: torch.Tensor,
                   shapes: Sequence[Sequence[int]]
                   ) -> List[torch.Tensor]:
        raise NotImplementedError()

    def _pre_all_gather(self, send_tensor, shapes):
        shapes = list(map(tuple, shapes))
        assert len(shapes) == self.world_size
        assert shapes[self.rank] == send_tensor.shape

        if not send_tensor.is_contiguous():
            # Ensure contiguousness for the last time.
            logger.debug(
                'Doing all_gather on a non-contiguous tensor')
            send_tensor = send_tensor.contiguous()

        return (self, send_tensor, shapes), {}

    def gather(self, dst: int, send_tensor: torch.Tensor
               ) -> Optional[List[torch.Tensor]]:
        """
        Gather a list of tensors into a single process.

        This collective function is compile-time only,
        processes should interchange shape/dtype info internally.
        TODO other collective functions like broadcast should encapsulate the
        interchange too.
        """
        raise NotImplementedError()

    def _pre_gather(self, dst, send_tensor):
        assert 0 <= dst < self.world_size

        if not send_tensor.is_contiguous():
            # Ensure contiguousness for the last time.
            logger.debug(
                'Doing all_gather on a non-contiguous tensor')
            send_tensor = send_tensor.contiguous()
        return (self, dst, send_tensor), {}

    def gather_object_list(self, dst: int, obj: _T) -> Optional[List[_T]]:
        raise NotImplementedError()

    def _pre_gather_object_list(self, dst, obj):
        assert 0 <= dst < self.world_size
        return (self, dst, obj), {}

    @typing.overload
    def scatter(self, src: int) -> torch.Tensor: ...

    @typing.overload
    def scatter(
        self, src: int, tensors: Sequence[torch.Tensor]
    ) -> torch.Tensor: ...

    def scatter(
        self, src: int, tensors: Optional[Sequence[torch.Tensor]] = None
    ) -> torch.Tensor:
        raise NotImplementedError()

    def _pre_scatter(self, src, tensors=None):
        assert 0 <= src < self.world_size

        if src == self.rank:
            assert tensors is not None
            assert len(tensors) == self.world_size

            dtypes = set(t.dtype for t in tensors)
            assert len(dtypes) == 1

            tensors = list(tensors)  # copy the list
            for i, tensor in enumerate(tensors):
                if not tensor.is_contiguous():
                    # Ensure contiguousness for the last time.
                    logger.debug(
                        f'The {i}-th tensor to scatter is non-contiguous'
                    )
                    tensors[i] = tensor.contiguous()
        else:
            tensors = None  # unused

        return (self, src, tensors), {}

    @typing.overload
    def scatter_object_list(self, src: int) -> Any: ...
    @typing.overload
    def scatter_object_list(self, src: int, objs: List[_T]) -> _T: ...

    def scatter_object_list(
        self, src: int, objs: Optional[List[_T]] = None
    ) -> _T:
        raise NotImplementedError()

    def _pre_scatter_object_list(self, src, objs=None):
        assert 0 <= src < self.world_size

        if src == self.rank:
            assert objs is not None
            assert len(objs) == self.world_size
        else:
            objs = None  # unused

        return (self, src, objs), {}

    def barrier(self):
        raise NotImplementedError()

    # def wait_all(self, works: Sequence['Work']):
    #     for work in works:
    #         work.wait()

    #     # TODO for CUDA, `wait()` only waits until the receiving command
    #     # is pushed to CUDA stream, the data is far from being ready,
    #     # in order to profile the communication overheads, we need to do
    #     # CUDA-specific synchronization.
    #     # P.S. such CUDA-stream completion is enough for correctness with CUDA
    #     # since CUDA is itself asynchronous.

    #     # if self.device.type == 'cuda':
    #     #     torch.cuda.synchronize()


class DummyDistEnv(DistEnv):
    def __init__(self, device_type: Literal['cpu', 'cuda']) -> None:
        if device_type == 'cpu':
            comm_device = torch.device('cpu')
        else:
            comm_device = torch.device(device_type, 0)
            torch.cuda.set_device(comm_device)

        super().__init__(world_size=1, rank=0, local_rank=0, device=comm_device)

    def broadcast(self, src: int, tensor: Optional[torch.Tensor] = None,
                  *,
                  shape: Optional[Sequence[int]] = None,
                  dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        assert src == 0, \
            "Broadcast in dummy environment is from rank 0 to rank 0"
        assert tensor is not None, \
            "Broadcast sender in dummy environment must provide the tensor"
        return tensor.clone()

    def broadcast_object_list(self, src: int,
                              object_list: Optional[list] = None
                              ) -> list:
        assert src == 0, \
            "Broadcast in dummy environment is from rank 0 to rank 0"
        assert isinstance(object_list, list)
        return copy.deepcopy(object_list)

    def def_isend(self, tensor: torch.Tensor, dst: int, tag: int) -> Any:
        assert False, "peer-to-peer API shouldn't be called"
        return None

    def def_irecv(self, buffer: torch.Tensor, src: int, tag: int) -> Any:
        assert False, "peer-to-peer API shouldn't be called"
        return None

    def batch_isend_irecv(self, p2p_ops: List[Any]) -> List[Any]:
        assert not p2p_ops, "Cannot have any P2P communication"
        return []

    def all_to_all_single(self, local_input: torch.Tensor) -> torch.Tensor:
        return local_input.clone()

    def all_to_all(self, tensors: Sequence[torch.Tensor]
                   ) -> List[torch.Tensor]:
        return [tensors[0].clone()]

    def all_gather_into_tensor(self, send_tensor: torch.Tensor,
                               form: Literal['concat', 'stack'] = 'concat'):
        if form == 'concat':
            return send_tensor.clone()
        else:
            return send_tensor[None, ...].clone()

    def all_gather(self, send_tensor: torch.Tensor,
                   shapes: Sequence[Sequence[int]]
                   ) -> List[torch.Tensor]:
        return [send_tensor.clone()]

    def gather(self, dst: int, send_tensor: torch.Tensor
               ) -> Optional[List[torch.Tensor]]:
        return [send_tensor.clone()]

    def gather_object_list(self, dst: int, obj: _T) -> Optional[List[_T]]:
        return [copy.deepcopy(obj)]

    def scatter(
        self, src: int, tensors: Optional[Sequence[torch.Tensor]]
    ) -> torch.Tensor:
        assert tensors is not None
        return tensors[0].clone()

    def scatter_object_list(
        self, src: int, objs: Optional[List[_T]] = None
    ) -> _T:
        assert objs is not None
        return copy.deepcopy(objs[0])

    def abort(self):
        exit(-1)

    def barrier(self):
        return None


class TorchDistEnv(DistEnv):
    """
    Note:
    NCCL backend does not support CPU communication.
    We'd better reject backend=NCCL + device_type=CPU combination for JIT.

    Also, GLOO backend (via TorchGlooDistEnv) does not support CUDA
    communication well.
    We'd better reject backend=GLOO + device_type=CUDA combination too.
    """

    def __init__(
        self, world_size: int, rank: int, local_rank: int, device: torch.device
    ) -> None:
        super().__init__(world_size, rank, local_rank, device)

    def broadcast(self, src: int, tensor: Optional[torch.Tensor] = None,
                  *,
                  shape: Optional[Sequence[int]] = None,
                  dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if src == self.rank:
            assert tensor is not None
            dist.broadcast(tensor, src)
            return tensor
        else:
            assert shape is not None and dtype is not None
            recv_buffer = torch.empty(shape, dtype=dtype,
                                      device=self.comm_device)
            dist.broadcast(recv_buffer, src)
            return recv_buffer

    def broadcast_object_list(self, src: int,
                              object_list: Optional[list] = None,
                              ) -> list:
        """
        Remark:
        When a CUDA tensor is a part of the broadcasted object, torch.dist
        will try to reconstruct it on the destination worker using
        literally the same Tensor.device, such as 'cuda:3'. Such a torch.device
        may not exist on that worker.

        To eliminate such risks, when broadcasting tensors as _objects_,
        ensure those tensors are with `device='cpu'`.
        """
        if src == self.rank:
            dist.broadcast_object_list(
                [object_list], src, device=self.comm_device
            )
            return object_list  # type: ignore
        else:
            recv_list = [None]
            dist.broadcast_object_list(recv_list, src, device=self.comm_device)
            return recv_list[0]  # type: ignore

    def def_isend(self, tensor: torch.Tensor, dst: int, tag: int) -> dist.P2POp:
        return dist.P2POp(dist.isend, tensor, peer=dst, tag=tag)

    def def_irecv(self, buffer: torch.Tensor, src: int, tag: int) -> dist.P2POp:
        return dist.P2POp(dist.irecv, buffer, peer=src, tag=tag)

    def batch_isend_irecv(self, p2p_ops: List[dist.P2POp]) -> List[dist.Work]:
        if p2p_ops:
            # NOTE an extra len!=0 check is needed here,
            # as `dist.batch_isend_irecv` would raise if the list is empty.
            return dist.batch_isend_irecv(p2p_ops)
        else:
            return []

    def all_to_all_single(self, local_input: torch.Tensor) -> torch.Tensor:
        local_output = torch.empty_like(local_input)
        dist.all_to_all_single(local_output, local_input)
        return local_output

    def all_to_all(self, tensors: Sequence[torch.Tensor]
                   ) -> List[torch.Tensor]:
        """
        Currently the only use case is during JIT time, and all input tensors
        have ndim==1.
        """
        dtype = tensors[0].dtype

        send_lengths = torch.tensor([t.shape[0] for t in tensors],
                                    dtype=torch.int64, device=self.comm_device)
        recv_lengths = self.all_to_all_single(send_lengths)

        buffers = [
            torch.empty((recv_length,), dtype=dtype, device=self.comm_device)
            for recv_length in recv_lengths.tolist()
        ]
        dist.all_to_all(buffers, tensors)

        return buffers

    def all_gather_into_tensor(self, send_tensor: torch.Tensor,
                               form: Literal['concat', 'stack'] = 'concat'):
        """
        torch.distributed doc says all tensor sizes must be the same,
        Currently we only use all_gather_into_tensor for runtime aggregators,
        we explicitly exclude cases of different sizes.
        """
        shape = list(send_tensor.shape)

        if shape[0] != 1:
            raise NotImplementedError("Support different tensor sizes")

        if form == 'concat':
            shape[0] = shape[0] * self.world_size
        else:
            shape.insert(0, self.world_size)

        recv_buffer = torch.empty(
            shape, dtype=send_tensor.dtype, device=self.comm_device)

        dist.all_gather_into_tensor(recv_buffer, send_tensor)
        return recv_buffer

    def all_gather(
        self, send_tensor: torch.Tensor,
        # NOTE base method has `shape: Seq[Seq[int]]` but here we have made it
        # `shape: List[Tuple[int]]` by the _pre_all_gather filter.
        shapes: Sequence[Sequence[int]]
    ) -> List[torch.Tensor]:
        """
        torch.distributed doc does not say it, but GLOO and MPI backends
        do not support all_gather with different shapes (even batch sizes).
        Use broadcast to implement.

        This method is used in Tensor.collect.
        """
        recv_buffers = []
        for i in range(self.world_size):
            if i == self.rank:
                self.broadcast(src=i, tensor=send_tensor)
                recv_buffers.append(send_tensor.clone())
            else:
                recv = self.broadcast(src=i, shape=shapes[i],
                                      dtype=send_tensor.dtype)
                recv_buffers.append(recv)
        return recv_buffers

    def gather(
        self, dst: int, send_tensor: torch.Tensor
    ) -> Optional[List[torch.Tensor]]:
        """
        torch.distributed doc says all tensor sizes must be the same,
        in EASIER cases the sizes are generally different.
        Use P2P to reimplement.

        P.S. NCCL can still take different shapes however GLOO cannot.
        """
        shape = tuple(send_tensor.shape)

        # But for current use cases we still have the same subshapes
        from easier.core.runtime.utils import check_collective_equality
        check_collective_equality("subshape", shape[1:], dist_env=self)

        if self.rank == dst:
            recv_buffers = []
            for w in range(self.world_size):
                if w != self.rank:
                    batch_size_w = self.recv_int64(w, w)

                    buffer = torch.empty(
                        (batch_size_w,) + shape[1:],
                        dtype=send_tensor.dtype,
                        device=self.comm_device
                    )
                    recv_buffers.append(buffer)
                    # since we are sending/recving twice per node,
                    # TODO we'd use blocking recv here.
                    self.recv(buffer, w, w)

                else:
                    recv_buffers.append(send_tensor.clone())

            return recv_buffers

        else:
            self.send_int64(shape[0], dst, self.rank)
            self.send(send_tensor, dst, self.rank)
            return None

    def gather_object_list(self, dst: int, obj: _T) -> Optional[List[_T]]:
        """
        torch.distributed.gather_object_list does not take device parameter,
        we should pickle the obj manually.
        """
        u8 = self._pickle_obj(obj)
        u8s = self.gather(dst, u8)

        if self.rank == dst:
            assert u8s is not None
            recvs = [self._unpickle_obj(u8) for u8 in u8s]
        else:
            recvs = None
        return recvs

    def _pickle_obj(self, obj):
        """
        Serialize the object into the comm device
        """
        u8 = torch.from_numpy(
            # numpy.frombuffer creates a view, without taking ownership of the
            # memory, and will trigger non-writable warning
            # during torch.from_numpy. copy() to deprecate the warning.
            numpy.frombuffer(pickle.dumps(obj), dtype=numpy.uint8).copy()
        ).to(device=self.comm_device)
        return u8

    def _unpickle_obj(self, u8_tensor: torch.Tensor):
        assert u8_tensor.dtype == torch.uint8
        obj = pickle.loads(
            u8_tensor.cpu().numpy().tobytes()
        )
        return obj

    def scatter(
        self, src: int, tensors: Optional[Sequence[torch.Tensor]]
    ) -> torch.Tensor:
        """
        torch.distributed doc says all tensor sizes must be the same,
        NCCL can take different shapes but GLOO cannot. Use P2P to unify.
        """
        if self.rank == src:
            assert tensors is not None
            shapes_dtypes = [(t.shape, t.dtype) for t in tensors]

            # Lightweight objects, just broadcast.
            self.broadcast_object_list(src, shapes_dtypes)

            # But for current use cases we still have the same subshapes
            assert len(set(t.shape[1:] for t in tensors)) == 1

            ops = []
            for w in range(self.world_size):
                if w != self.rank:
                    isend = self.def_isend(tensors[w], w, w)
                    ops.append(isend)
            for req in self.batch_isend_irecv(ops):
                req.wait()

            return tensors[src].clone()

        else:
            shape, dtype = self.broadcast_object_list(src)[self.rank]
            buffer = self.recv(torch.empty(
                shape, dtype=dtype, device=self.comm_device
            ), src=src, tag=self.rank)
            return buffer

    def scatter_object_list(
        self, src: int, objs: Optional[List[_T]] = None
    ) -> _T:
        """
        torch.distributed.scatter_object_list does not take device parameter,
        we should pickle the obj manually.
        """
        if self.rank == src:
            assert objs is not None
            u8s = [self._pickle_obj(obj) for obj in objs]
            self.scatter(src, u8s)
            u8 = u8s[self.rank]
        else:
            u8 = self.scatter(src, None)

        obj = self._unpickle_obj(u8)
        return obj

    def barrier(self):
        dist.barrier()


class TorchDistGlooDistEnv(TorchDistEnv):
    def all_to_all(self, tensors: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """
        GLOO doesn't support all to all.
        """
        dtype = tensors[0].dtype
        send_lengths = torch.tensor([t.shape[0] for t in tensors],
                                    dtype=torch.int64, device=self.comm_device)
        recv_lengths = self.all_to_all_single(send_lengths)

        p2ps = []

        for u in range(self.world_size):
            if u != self.rank:
                if send_lengths[u] > 0:
                    isend_op = self.def_isend(tensors[u], u, tag=u)
                    p2ps.append(isend_op)

        buffers: List[torch.Tensor] = []
        for w in range(self.world_size):
            if w != self.rank:
                recv_length = int(recv_lengths[w])
                buffer = torch.empty(
                    (recv_length,), dtype=dtype, device=self.comm_device)
                buffers.append(buffer)

                if recv_length > 0:
                    irecv_op = self.def_irecv(buffer, w, tag=self.rank)
                    p2ps.append(irecv_op)
            else:
                buffers.append(tensors[self.rank].clone())

        for p2p in self.batch_isend_irecv(p2ps):
            p2p.wait()

        return buffers


class TorchDistMpiDistEnv(TorchDistEnv):
    def all_gather_into_tensor(
        self,
        send_tensor: torch.Tensor,
        form: Literal['concat', 'stack'] = 'concat'
    ):
        """
        torch.distributed.all_gather_into_tensor API is not supported by
        the MPI backend. Use all_gather to implement (causing unnecessry
        split and concat).
        Currently we only use all_gather_into_tensor for runtime aggregators,
        we explicitly exclude cases of different sizes.
        """
        shape = list(send_tensor.shape)

        if shape[0] != 1:
            raise NotImplementedError("Support different tensor sizes")

        recv_buffers = [
            torch.empty(
                shape, dtype=send_tensor.dtype, device=self.comm_device
            ) for w in range(self.world_size)
        ]

        # In our use cases of aggregators all input tensors have the same size.
        # Don't call self.all_gather because we have reimplemented it
        # using broadcast in base TorchDistEnv.
        # Just call the slightly faster dist.all_gather.
        dist.all_gather(recv_buffers, send_tensor)

        if form == 'concat':
            return torch.concat(recv_buffers)
        else:
            return torch.stack(recv_buffers)

    def def_isend(self, tensor: torch.Tensor, dst: int, tag: int) -> Any:
        return (dist.isend, tensor, dst, tag)

    def def_irecv(self, buffer: torch.Tensor, src: int, tag: int) -> Any:
        return (dist.irecv, buffer, src, tag)

    def batch_isend_irecv(self, p2p_ops: List[tuple]) -> List[dist.Work]:
        """
        We override def_isend/def_irecv/batch_isend_irecv mainly for
        CUDA cases in MPI, where torch.distributed.batch_isend_irecv
        tries to group all P2P requests.
        torch.distributed MPI backend does not support grouping.

        Now we simply keep the definition-only semantics of def_isend/irecv
        and invoke all at once here.

        TODO However, MPI supports both CPU/GPU ops, including P2P,
        (if it's CUDA-aware and all_gather_into_tensor fix is still needed)
        unlike GLOO/NCCL doesn't support GPU/CPU P2P.
        """
        works = []
        for tp in p2p_ops:
            op, tensor, peer, tag = tp
            work = op(tensor, peer, tag=tag)
            works.append(work)
        return works


# must be explicitly supported by EASIER
DistBackendStr: TypeAlias = Literal['gloo', 'nccl', 'mpi']


class CommBackendConfig:
    def __init__(
        self,
        comm_backend_config_str: str
    ) -> None:
        self._config: Union[DistBackendStr, Dict[str, DistBackendStr]]

        def _check_backend(x) -> DistBackendStr:
            if x not in ['gloo', 'nccl', 'mpi']:
                raise EasierJitException(
                    f"Unsupported communication backend {x}"
                )
            return x

        if ',' not in comm_backend_config_str:
            self._config = _check_backend(comm_backend_config_str)

        else:
            self._config = {}

            def _bad_format():
                raise EasierJitException(
                    f"Communication backend '{comm_backend_config_str}'"
                    " is bad-format"
                )
            for channel in comm_backend_config_str.split(','):
                kv = channel.split(':')
                if len(kv) == 2:
                    device_type, backend = kv
                    backend = _check_backend(backend)
                    self._ensure_known_backend_devicetype_compatibility(
                        backend, device_type
                    )
                    self._config[device_type] = backend

                else:
                    _bad_format()
            else:
                _bad_format()

    def get_backends(self) -> List[DistBackendStr]:
        if isinstance(self._config, str):
            return [self._config]
        else:
            return list(self._config.values())

    def backend_specific_setup(self):
        for backend in self.get_backends():
            if backend == 'nccl':
                # Validate if CUDA is available and CUDA device number
                # NCCL requires each rank has an individual card.
                if not torch.cuda.is_available():
                    raise EasierJitException(
                        "The communication backend 'nccl' is specified"
                        " but CUDA is not available"
                    )

                if self.get_local_rank() >= torch.cuda.device_count():
                    raise EasierJitException(
                        "The communication backend 'nccl' requires each process"
                        " has a dedicated CUDA device. This machine has only"
                        f" {torch.cuda.device_count()} CUDA device(s).\n"
                        "If CUDA devices are not the intended computing"
                        "devices, consider explicitly specifying communication"
                        " backends excluding 'nccl' to `easier.init()`"
                    )

                # Although EASIER always explicitly specifies the CUDA index,
                # if users call pickle-based torch.distributed APIs, this is
                # required, (only NCCL needs, CUDA-aware MPI does not)
                # and this can help avoid CUDA:0 OOM.
                #
                # TODO benefits the CUDA MPI, but since MPI is not strongly
                # bound with CUDA, we need to avoid enforcing dedicated
                # CUDA devices in case users do not intend to use CUDA at all.
                if torch.cuda.is_available():
                    cuda_device = torch.device('cuda', self.get_local_rank())
                    logger.info(f"Set default CUDA device: {cuda_device}")
                    torch.cuda.set_device(cuda_device)

            if backend == 'mpi':
                if not torch.distributed.is_mpi_available():
                    raise EasierJitException(
                        "The communication backend 'mpi' is not supported by"
                        " the PyTorch package"
                    )

    def _ensure_known_backend_devicetype_compatibility(
        self, comm_backend, device_type
    ) -> None:
        """
        Check if a single combination is known to be incompatible.

        However, we don't know the compatibility for all kinds of device types,
        especially for 'torch' compilation backend,
        and we don't know if MPI is CUDA-aware.
        """
        if (comm_backend, device_type) in [
            ('gloo', 'cuda'),  # No P2P ops
            ('nccl', 'cpu')    # not supported at all
        ]:
            raise EasierJitException(
                f"EASIER does not support use the device type '{device_type}'"
                f" with the communication backend '{comm_backend}'"
            )

    def ensure_device_type_compatibility(self, device_type: str) -> None:
        if isinstance(self._config, str):
            self._ensure_known_backend_devicetype_compatibility(
                self._config, device_type
            )
        else:
            if device_type not in self._config.keys():
                raise EasierJitException(
                    f"No communication backend is specified for {device_type}"
                )

    def warn_if_device_type_is_unknown(self, device_type: str):
        # TODO although torch.distributed supports custom added backends,
        # EASIER cannot directly use them due to the inconsistent supports on
        # each comm API, therefore we don't actually allow devices beyond
        # what are supported by GLOO/NCCL/MPI backends, that are: CUDA & CUDA.
        if device_type not in ['cpu', 'cuda']:
            logger.warning(
                f"The device type {device_type} is not known by EASIER,"
                " please ensure the data are properly assigned to each device"
            )

    @functools.cache
    def get_local_rank(self) -> int:
        # If MPI is mentioned, it must be launched by MPIRUN.
        #
        # Nonetheless, it's possible to use MPIRUN and use `nccl`,
        # but users need to set WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR etc.
        # to reconstruct the environment that `torch.distributed + nccl` need.
        if 'mpi' in self.get_backends():
            local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        else:
            local_rank = int(os.environ['LOCAL_RANK'])

        return local_rank

    def get_torch_dist_backend_for(self, device_type: str) -> DistBackendStr:
        if isinstance(self._config, str):
            self._ensure_known_backend_devicetype_compatibility(
                self._config, device_type
            )
            return self._config
        else:
            return self._config[device_type]

    def get_default_device_type(self) -> str:
        if isinstance(self._config, str):
            if self._config == 'nccl':
                return 'cuda'
            else:
                return 'cpu'
        else:
            # TODO by prioritizing checking CUDA, we are actually trying to
            # leverage CUDA in the pre-AOT and AOT stages.
            # And we are sure CUDA is capable for vectorized ops in AOT
            # (but still may CUDA OOM)
            # TODO make AOT passes retrieve the default DistEnv
            for default_device_type in ['cuda', 'cpu']:
                if default_device_type in self._config:
                    return default_device_type

            default_device_type = next(iter(self._config.keys()))
            return default_device_type

    def __str__(self) -> str:
        if isinstance(self._config, str):
            return self._config
        else:
            return ','.join(f'{k}:{v}' for k, v in self._config.items())


_comm_backend_config: Optional[CommBackendConfig] = None

# whatever device type, may be other than 'cpu' and 'cuda'
_runtime_device_type: Optional[str] = None

# Keys are (backend, devicetype) tuples
_dist_envs: Dict[Tuple[str, str], DistEnv] = {}


def set_dist_env_runtime_backend_config(
    comm_backend_config_str: str
) -> CommBackendConfig:
    # TODO will data-to-send still be too big for CUDA memory, even though
    # we move AOT-compilation data back to CPU each time after comm?
    global _comm_backend_config
    assert _comm_backend_config is None
    _comm_backend_config = CommBackendConfig(comm_backend_config_str)

    _comm_backend_config.backend_specific_setup()

    return _comm_backend_config


def set_dist_env_runtime_device_type(
    comm_device_type: str
) -> None:
    if _comm_backend_config is None:
        raise EasierJitException(
            "The runtime communication backend isn't set,"
            " please ensure `easier.init()` has been called properly."
        )

    _comm_backend_config.ensure_device_type_compatibility(comm_device_type)
    _comm_backend_config.warn_if_device_type_is_unknown(comm_device_type)

    global _runtime_device_type
    # TODO leaving this unchecked enable multiple runs of esr.compile
    # assert _runtime_device_type is None
    _runtime_device_type = comm_device_type


def _get_or_init_dist_env(device_type: str) -> DistEnv:
    if _comm_backend_config is None:
        raise EasierJitException("The backend for runtime isn't set")

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = _comm_backend_config.get_local_rank()
    comm_device = torch.device(device_type, local_rank)

    comm_backend = _comm_backend_config.get_torch_dist_backend_for(device_type)

    key = (comm_backend, device_type)
    if key not in _dist_envs:
        if comm_backend == 'gloo':
            dist_env_cls = TorchDistGlooDistEnv
        elif comm_backend == 'mpi':
            dist_env_cls = TorchDistMpiDistEnv
        else:
            dist_env_cls = TorchDistEnv

        dist_env = dist_env_cls(world_size, rank, local_rank, comm_device)
        _dist_envs[key] = dist_env

    return _dist_envs[key]


def get_default_dist_env() -> DistEnv:
    """
    Get a working DistEnv on the default device of the comm backend.
    Typically used when the device of the most performance is not decided yet
    or may not exist.
    """
    if _comm_backend_config is None:
        raise EasierJitException(
            "The runtime communication backend isn't set,"
            " please ensure `easier.init()` has been called properly."
        )

    device_type = _comm_backend_config.get_default_device_type()
    return _get_or_init_dist_env(device_type)


def get_runtime_dist_env() -> DistEnv:
    """
    Get the DistEnv instance for communication during JIT runtime.
    This instance can be retrieved only during or after JIT process.
    """
    if _comm_backend_config is None:
        raise EasierJitException(
            "The runtime communication backend isn't set,"
            " please ensure `easier.init()` has been called properly."
        )
    if _runtime_device_type is None:
        raise EasierJitException("The device type for runtime isn't set")
    return _get_or_init_dist_env(_runtime_device_type)
