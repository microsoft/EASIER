# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast
from typing_extensions import Literal, TypeAlias, TypeVar
import typing
import threading
import time
import functools
import copy

import torch
import torch.distributed as dist

from easier.core.utils import logger, EasierJitException, LOGGING_TRACE

_T = TypeVar("_T")


def _wrap_commapi_pre_filter(prefilter, api):
    """
    Both `api` and `pre_filter` function objects are not 
    bound to some DistEnv instance yet, the `self` argument
    will be included at the head in `args` in the wrapper.
    """
    @functools.wraps(api)
    def wrapper(*args, **kwargs):
        if logger.level <= LOGGING_TRACE:
            this = args[0].__class__.__name__
            logger.log(
                LOGGING_TRACE,
                f'{this}.{api.__name__}(*{args[1:]}, **{kwargs})'
            )

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

    def gather_object(self, dst: int, obj: _T) -> Optional[List[_T]]:
        raise NotImplementedError()

    def _pre_gather_object(self, dst, obj):
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
    def scatter_object(self, src: int) -> Any: ...
    @typing.overload
    def scatter_object(self, src: int, objs: List[_T]) -> _T: ...

    def scatter_object(self, src: int, objs: Optional[List[_T]] = None) -> _T:
        raise NotImplementedError()

    def _pre_scatter_object(self, src, objs=None):
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

    def gather_object(self, dst: int, obj: _T) -> Optional[List[_T]]:
        return [copy.deepcopy(obj)]

    def scatter(
        self, src: int, tensors: Optional[Sequence[torch.Tensor]]
    ) -> torch.Tensor:
        assert tensors is not None
        return tensors[0].clone()

    def scatter_object(self, src: int, objs: Optional[List[_T]] = None) -> _T:
        assert objs is not None
        return copy.deepcopy(objs[0])

    def abort(self):
        exit(-1)

    def barrier(self):
        return None


class TorchDistEnv(DistEnv):
    def __init__(
        self,
        device_type: str,
        backend: Literal['gloo', 'nccl', 'mpi'],
        torch_dist_init_kwargs={}
    ) -> None:
        if backend in ['gloo', 'nccl']:
            local_rank = int(os.environ['LOCAL_RANK'])
        elif backend in ['mpi']:
            local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        else:
            raise EasierJitException(f"Unknown backend {backend}")

        comm_device = torch.device(device_type, local_rank)
        if device_type == 'cuda':
            logger.info(f"Set default CUDA device: {comm_device}")
            torch.cuda.set_device(comm_device)

        if dist.is_initialized():
            logger.info(
                "The user has initialized torch.distributed. Please ensure"
                " torch.distributed communication backend is compatible with"
                " EASIER compilation backend"
            )
        else:
            logger.info("Initializing torch.distributed")
            dist.init_process_group(backend, **torch_dist_init_kwargs)

        # the `get_backend_config()` may return like "cpu:gloo,cuda:nccl"
        logger.info(
            f"torch.distributed"
            f" backend={dist.get_backend_config()} rank={dist.get_rank()}"
            f" local_rank={local_rank}"
        )

        super().__init__(
            world_size=dist.get_world_size(),
            rank=dist.get_rank(),
            local_rank=local_rank,
            device=comm_device
        )

        self.backend = backend


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
            dist.broadcast_object_list([object_list], src)
            return object_list  # type: ignore
        else:
            recv_list = [None]
            dist.broadcast_object_list(recv_list, src)
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
        recv_buffers = [
            torch.empty(shape, dtype=send_tensor.dtype,
                        device=self.comm_device)
            for shape in shapes
        ]
        dist.all_gather(recv_buffers, send_tensor)
        return recv_buffers

    def gather(
        self, dst: int, send_tensor: torch.Tensor
    ) -> Optional[List[torch.Tensor]]:
        """
        torch.distributed doc says all tensor sizes must be the same,
        NCCL can take different shapes but GLOO cannot. Use P2P to unify.
        """
        shape = tuple(send_tensor.shape)
        shapes = self.gather_object(dst, shape)

        if self.rank == dst:
            assert shapes is not None
            recv_buffers = []
            ops = []
            for w in range(self.world_size):
                if w != self.rank:
                    buffer = torch.empty(
                        shapes[w],
                        dtype=send_tensor.dtype,
                        device=self.comm_device
                    )
                    recv_buffers.append(buffer)

                    irecv = self.def_irecv(buffer, w, w)
                    ops.append(irecv)
                else:
                    recv_buffers.append(send_tensor)
            for req in self.batch_isend_irecv(ops):
                req.wait()
            
            return recv_buffers

        else:
            self.send(send_tensor, dst, self.rank)
            return None


    def gather_object(self, dst: int, obj: _T) -> Optional[List[_T]]:
        if self.rank == dst:
            recvs = [None] * self.world_size  # type: ignore
        else:
            # in this case it must be None as torch.dist.gather_object requires
            recvs = None   # type: ignore

        dist.gather_object(obj, recvs, dst)

        return recvs  # type: ignore

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
            self.scatter_object(src, shapes_dtypes)

            ops = []
            for w in range(self.world_size):
                if w != self.rank:
                    isend = self.def_isend(tensors[w], w, w)
                    ops.append(isend)
            for req in self.batch_isend_irecv(ops):
                req.wait()
            
            return tensors[src]
        
        else:
            shape, dtype = self.scatter_object(src)
            buffer = self.recv(torch.empty(
                shape, dtype=dtype, device=self.comm_device
            ), src=src, tag=self.rank)
            return buffer

    def scatter_object(self, src: int, objs: Optional[List[_T]] = None) -> _T:
        recvs = [None]
        dist.scatter_object_list(recvs, objs, src=src)
        return recvs[0]  # type: ignore

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
                buffers.append(tensors[self.rank])

        for p2p in self.batch_isend_irecv(p2ps):
            p2p.wait()

        return buffers

    def all_gather(
        self, send_tensor: torch.Tensor, shapes: Sequence[Sequence[int]]
    ) -> List[torch.Tensor]:
        """
        torch.distributed doc does not say it,
        but GLOO backend doesn't support all_gather with different shapes.
        Use broadcast to implement.
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
    
class TorchDistMpiDistEnv(TorchDistEnv):
    def all_gather_into_tensor(
        self,
        send_tensor: torch.Tensor,
        form: Literal['concat', 'stack'] = 'concat'
    ):
        """
        torch.distributed.all_gather_into_tensor API is not supported by
        the MPI backend. Use all_gather to implement.
        Currently we only use all_gather_into_tensor for runtime aggregators,
        we explicitly exclude cases of different sizes.
        """
        shape = list(send_tensor.shape)

        if shape[0] != 1:
            raise NotImplementedError("Support different tensor sizes")

        # In our use cases of aggregators all input tensors have the same size.
        tensors = self.all_gather(send_tensor, [shape] * self.world_size)

        if form == 'concat':
            return torch.concat(tensors)
        else:
            return torch.stack(tensors)

    def batch_isend_irecv(self, p2p_ops: List[dist.P2POp]) -> List[dist.Work]:
        """
        MPI backend invokes send and recv immediately,
        and will raise if dist.batch_isend_irecv is called.
        """
        return []
    


_runtime_devicetype_backend: Optional[Tuple[
    str,  # whatever device type, may be other than 'cpu' and 'cuda'
    Literal['gloo', 'nccl', 'mpi']  # must be explicitly supported by EASIER
]] = None

_dist_envs: Dict[Tuple[str, str], DistEnv] = {}

def config_runtime_dist_env(
    comm_device_type: str,
    comm_backend: Literal['gloo', 'nccl', 'mpi', None]
):
    """
    If `comm_backend_type` is not specified, infer from the environment:

    -   'gloo' for 'cpu', 'nccl' for 'cuda';

    If users would like to use MPIRUN plus `torch.distributed` built from
    source with (CUDA-aware) MPI support,
    they should specify 'mpi' explicitly.

    P.S. it's possible to use MPIRUN but use `nccl` instead,
    if users set WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR MASTER_PORT etc.
    to reconstruct the environment that `torch.distributed + nccl` needs.

    TODO will data-to-send still be too big for CUDA memory, even though
    we move AOT-compilation data back to CPU each time after comm?
    """
    if comm_backend is None:
        if comm_device_type == 'cpu':
            comm_backend = 'gloo'
        elif comm_device_type == 'cuda':
            comm_backend = 'nccl'
        else:
            raise EasierJitException(
                "Cannot device communication backend for device_type"
                f" {comm_device_type}, please specify `comm_backend` parameter"
                " or EASIER_COMM_BACKEND environment variable explicitly"
            )
    assert comm_backend is not None
    
    # DummyDistEnv is not a standard DistEnv for public use.
    global _runtime_devicetype_backend
    assert _runtime_devicetype_backend is None
    _runtime_devicetype_backend = (comm_device_type, comm_backend)

def get_runtime_dist_env() -> DistEnv:
    """
    Get the DistEnv instance for communication during JIT runtime.
    This instance can be retrieved only during or after JIT process.
    """
    if _runtime_devicetype_backend is None:
        raise EasierJitException("The backend for runtime isn't set")

    if _runtime_devicetype_backend not in _dist_envs:
        device_type, comm_backend = _runtime_devicetype_backend

        if comm_backend == 'gloo':
            dist_env = TorchDistGlooDistEnv(device_type, 'gloo')
        elif comm_backend == 'mpi':
            dist_env = TorchDistMpiDistEnv(device_type, 'mpi')
        else:
            dist_env = TorchDistEnv(device_type, comm_backend)

        _dist_envs[_runtime_devicetype_backend] = dist_env

    return _dist_envs[_runtime_devicetype_backend]