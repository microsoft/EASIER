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

from mpi4py import MPI

from easier.core.utils import logger, EasierJitException

_T = TypeVar("_T")


def is_launched_by_easier_launcher():
    return os.environ.get("EASIER_USE_EASIER_LAUNCHER", None) is not None


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
    def __init__(self,
                 world_size: int, rank: int, local_rank: int, host_rank: int,
                 device: Union[str, torch.device]) -> None:
        """
        Args:
        -   device: To serve as the channel for communication.
                Aligning with the target accelerator hardware can achieve
                minimum data movement overheads.
        """
        self.world_size = world_size
        self.rank = rank
        self.local_rank = local_rank
        self.host_rank = host_rank

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

    @property
    def is_host(self):
        return self.rank == self.host_rank

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
    def broadcast_object_list(self, src: int, object_list: List[Any]): ...
    @typing.overload
    def broadcast_object_list(self, src: int) -> List[Any]: ...

    def broadcast_object_list(self, src: int,
                              object_list: Optional[list] = None
                              ) -> Optional[list]:
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
                        f'The {i}-th tensor to scatter is non-contiguous')
                    tensors[i] = tensor.contiguous()
        else:
            objs = None  # unused

        return (self, src, objs), {}

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

    def abort(self):
        """
        Abort all processes.

        This method is not required to be called collectively.
        """
        raise NotImplementedError()

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

        super().__init__(world_size=1, rank=0, local_rank=0, host_rank=0,
                         device=comm_device)

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
                              ) -> Optional[list]:
        assert src == 0, \
            "Broadcast in dummy environment is from rank 0 to rank 0"
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
    def __init__(self, device_type: Literal['cpu', 'cuda'],
                 torch_dist_init_kwargs={}) -> None:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        host_rank = 0

        if device_type == 'cpu':
            comm_device = torch.device('cpu')
        else:
            comm_device = torch.device(device_type, local_rank)
            torch.cuda.set_device(comm_device)

        super().__init__(world_size, rank, local_rank, host_rank, comm_device)

        # TODO as we have enabled MPI, consider remove GLOO backend
        backend = 'nccl' if device_type == 'cuda' else 'gloo'
        self.backend = backend
        dist.destroy_process_group

        dist.init_process_group(backend, **torch_dist_init_kwargs)
        logger.info(
            f"Init torch.distributed "
            f"backend={backend} rank={rank} local_rank={local_rank}")

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
                              ) -> Optional[list]:
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
            return object_list
        else:
            recv_list = [None]  # type: ignore
            dist.broadcast_object_list(recv_list, src)
            return recv_list[0]

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

    def _gloo_all_to_all(self, tensors: Sequence[torch.Tensor],
                         send_lengths: torch.Tensor,
                         recv_lengths: torch.Tensor,
                         dtype) -> List[torch.Tensor]:
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

        if self.backend == 'gloo':
            return self._gloo_all_to_all(tensors, send_lengths, recv_lengths,
                                         dtype)

        buffers = [
            torch.empty((recv_length,), dtype=dtype, device=self.comm_device)
            for recv_length in recv_lengths.tolist()
        ]
        dist.all_to_all(buffers, tensors)

        return buffers

    def _gloo_all_gather_into_tensor(
        self, send_tensor: torch.Tensor,
        form: Literal['concat', 'stack'] = 'concat'
    ) -> torch.Tensor:
        # PyTorch GLOO communication backend doesn't support this primitive.
        tensors = [
            torch.empty_like(send_tensor, device=self.comm_device)
            for _ in range(self.world_size)
        ]
        dist.all_gather(tensors, send_tensor)

        if form == 'concat':
            return torch.concat(tensors)
        else:
            return torch.stack(tensors)

    def all_gather_into_tensor(self, send_tensor: torch.Tensor,
                               form: Literal['concat', 'stack'] = 'concat'):
        if self.backend == 'gloo':
            return self._gloo_all_gather_into_tensor(send_tensor, form)

        shape = list(send_tensor.shape)
        if form == 'concat':
            shape[0] = shape[0] * self.world_size
        else:
            shape.insert(0, self.world_size)

        recv_buffer = torch.empty(
            shape, dtype=send_tensor.dtype, device=self.comm_device)

        dist.all_gather_into_tensor(recv_buffer, send_tensor)
        return recv_buffer

    def _gloo_all_gather_different_shapes(
        self,
        send_tensor: torch.Tensor,
        shapes: List[Tuple[int, ...]]
    ):
        # PyTorch GLOO communication backend doesn't support all_gather with
        # different shapes.
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

    def all_gather(
        self, send_tensor: torch.Tensor,
        # NOTE base method has `shape: Seq[Seq[int]]` but here we have made it
        # `shape: List[Tuple[int]]` by the _pre_all_gather filter.
        shapes: List[Tuple[int, ...]]
    ) -> List[torch.Tensor]:
        if self.backend == 'gloo' and len(set(shapes)) > 1:
            return self._gloo_all_gather_different_shapes(
                send_tensor, shapes
            )

        recv_buffers = [
            torch.empty(shape, dtype=send_tensor.dtype,
                        device=self.comm_device)
            for shape in shapes
        ]
        dist.all_gather(recv_buffers, send_tensor)
        return recv_buffers

    def gather(self, dst: int, send_tensor: torch.Tensor
               ) -> Optional[List[torch.Tensor]]:
        shape = tuple(send_tensor.shape)
        shapes = self.gather_object(dst, shape)

        if self.rank == dst:
            assert shapes is not None
            recv_buffers = [
                torch.empty(shape, dtype=send_tensor.dtype,
                            device=self.comm_device)
                for shape in shapes
            ]
        else:
            recv_buffers = None

        dist.gather(send_tensor, recv_buffers, dst)

        if self.rank == dst:
            return recv_buffers
        else:
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
        if self.rank == src:
            assert tensors is not None
            [dtype] = self.broadcast_object_list(
                src, [tensors[0].dtype]
            )  # type: ignore
            shape = self.scatter_object(src, [t.shape for t in tensors])
        else:
            [dtype] = self.broadcast_object_list(src)  # type: ignore
            shape = self.scatter_object(src)  # type: ignore

        buffer = torch.empty(shape, dtype=dtype, device=self.comm_device)
        dist.scatter(buffer, tensors, src)

        return buffer

    def scatter_object(self, src: int, objs: Optional[List[_T]] = None) -> _T:
        recvs = [None]
        dist.scatter_object_list(recvs, objs, src=src)
        return recvs[0]  # type: ignore

    def abort(self):
        raise NotImplementedError("Consider aborting using CPU dist env.")

    def barrier(self):
        dist.barrier()


# TODO seems used in many place, move to utils.
def torch_dtype_to_numpy_dtype(torch_dtype: torch.dtype):
    return torch.empty([0], dtype=torch_dtype).numpy().dtype


class MPIDistEnv(DistEnv):
    def __init__(self, local_rank: Optional[int] = None) -> None:
        self.comm: MPI.Comm = get_mpi_communicator()

        if local_rank is None:
            # mpi4py.MPI.Comm does not have an API to get local rank,
            # so when read the local rank from OMPI mpirun env vars.
            # In cases that EASIER programs are not launched by mpirun,
            # mpi4py.MPI.Comm.WORLD still work (i.e. singleton world),
            # but the mpirun env vars does not exist, where we need to figure
            # out local rank ourselves.
            local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])

        super().__init__(self.comm.size, self.comm.rank,
                         local_rank=local_rank,
                         host_rank=0,
                         device='cpu')

        self._mpi_dtypes: Dict[torch.dtype, Optional[MPI.Datatype]] = {}

    def _check_unsupported_buffer_dtype(self, dtype: torch.dtype, method: str):
        """
        TODO MPI buffer-based APIs (capitalized APIs like Irecv) do not support
        torch.float16 etc. directly.
        We may need to bypass DLPack (if input is torch Tensor not numpy array)
        and specify it as uint8.
        """
        from mpi4py.util import dtlib

        if dtype not in self._mpi_dtypes:
            try:
                mpi_dtype = dtlib.from_numpy_dtype(
                    torch_dtype_to_numpy_dtype(dtype)
                )
            except ValueError:  # conversion fails
                mpi_dtype = None

            self._mpi_dtypes[dtype] = mpi_dtype

        unsupported = self._mpi_dtypes[dtype] is None
        if unsupported:
            raise NotImplementedError(
                f'MPI method {method} does not support dtype {dtype}'
            )

    def broadcast(self, src: int, tensor: Optional[torch.Tensor] = None,
                  *,
                  shape: Optional[Sequence[int]] = None,
                  dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if src == self.rank:
            assert tensor is not None

            self._check_unsupported_buffer_dtype(tensor.dtype, 'Bcast')

            # uppercase, buffer-based mpi4py op can be used on torch.Tensor,
            # via __dlpack__ protocol.
            self.comm.Bcast(tensor, src)
            return tensor
        else:
            assert shape is not None and dtype is not None

            self._check_unsupported_buffer_dtype(dtype, 'Bcast')

            recv_buffer = torch.empty(shape, dtype=dtype,
                                      device=self.comm_device)
            self.comm.Bcast(recv_buffer, src)  # uppercase, buffer-based
            return recv_buffer

    def broadcast_object_list(self, src: int,
                              object_list: Optional[list] = None,
                              ) -> Optional[list]:
        if src == self.rank:
            return self.comm.bcast(object_list, src)  # pickle-based
        else:
            object_list = None  # unused
            return self.comm.bcast(None, src)  # pickle-based

    # TODO MPIDistEnv.def_isend/def_irecv/batch_isend_irecv have very different
    # semantics than TorchDistEnv. As the abstraction of torch.distributed over
    # NCCL requires the non-blocking communication is defined-then-invoked
    # within a context of a "batch" (e.g. with NCCL these p2p ops become
    # a single ring-topo call).
    #
    # We definitely need to re-evaluate the network performance, and adapt
    # the DistEnv code for that. So for now let's simply invoke MPI p2p
    # immediately.

    def def_isend(self, tensor: torch.Tensor, dst: int, tag: int
                  ) -> MPI.Request:

        self._check_unsupported_buffer_dtype(tensor.dtype, 'Isend')

        return self.comm.Isend(tensor, dst, tag)

    def def_irecv(self, buffer: torch.Tensor, src: int, tag: int
                  ) -> MPI.Request:

        self._check_unsupported_buffer_dtype(buffer.dtype, 'Irecv')
        # mpi4py has very different usage on pickle-based recv method, where
        # the input must still be a memory buffer, but should be **big enough**
        # to store the pickle bytesstream.
        # TODO We may need to:
        # - cast float16 to float32 and send/recv;
        # - for torch Tensor, it's handled by DLPack protocol, where dtype is
        #   respected, we may "reinterpret cast" it to uint8 Tensor and Irecv.

        return self.comm.Irecv(buffer, src, tag)

    def batch_isend_irecv(self, p2p_ops: List[MPI.Request]
                          ) -> List[MPI.Request]:
        # TODO MPIDistEnv.batch_isend_irecv is a NO-OP.
        return p2p_ops

    def all_to_all_single(self, local_input: torch.Tensor) -> torch.Tensor:
        # We use mpi4py lowercase `alltoall` since `local_input` is a
        # small vector and cheap to pickle.
        return torch.tensor(self.comm.alltoall(local_input.tolist()))

    def all_to_all(self, tensors: Sequence[torch.Tensor]
                   ) -> List[torch.Tensor]:
        # TODO we use mpi4py lowercase, pickle-based `alltoall` to avoid
        # manage the concat-ed buffer for upper-case, buffer-based `Alltoallv`,
        # it's ok since this is only called JIT-time.
        return self.comm.alltoall(tensors)

    def all_gather_into_tensor(self, send_tensor: torch.Tensor,
                               form: Literal['concat', 'stack'] = 'concat'):

        self._check_unsupported_buffer_dtype(send_tensor.dtype, 'Allgather')

        shape = list(send_tensor.shape)
        if form == 'concat':
            shape[0] = shape[0] * self.world_size
        else:
            shape.insert(0, self.world_size)

        recv_buffer = torch.empty(
            shape, dtype=send_tensor.dtype, device=self.comm_device)

        # `all_gather_into_tensor` requires all send_tensors have the same
        # shape, as long as they are C-layout we don't need to specify the
        # mpi4py buffer spec.
        self.comm.Allgather(send_tensor, recv_buffer)

        return recv_buffer

    def all_gather(
        self, send_tensor: torch.Tensor,
        # NOTE base method has `shape: Seq[Seq[int]]` but here we have made it
        # `shape: List[Tuple[int]]` by the _pre_all_gather filter.
        shapes: List[Tuple[int, ...]]
    ) -> List[torch.Tensor]:
        # TODO `send_tensor` may have different batch sizes, we simply use
        # lowercase, pickle-baed `allgather` to do it. It's ok since
        # `dist_env.all_gather` is used only in `easier.Tensor.collect()`.
        recv_tensors = self.comm.allgather(send_tensor)
        return recv_tensors

    def gather(self, dst: int, send_tensor: torch.Tensor
               ) -> Optional[List[torch.Tensor]]:
        # TODO use pickle-based `comm.gather`.
        # Actually `dist_env.gather` is only used in `esr.Tensor.save()` where
        # it does not require to split the resultant buffer into tensors.
        # TODO send_tensor may have different batch sizes, use buffer-based
        # `comm.Gatherv`.
        return self.comm.gather(send_tensor, dst)

    def gather_object(self, dst: int, obj: _T) -> Optional[List[_T]]:
        return self.comm.gather(obj, dst)

    def scatter(
        self, src: int, tensors: Optional[Sequence[torch.Tensor]]
    ) -> torch.Tensor:
        # TODO remove MPIDistEnv, temporarily use pickle-based
        return self.comm.scatter(tensors, src)  # type: ignore

    def scatter_object(self, src: int, objs: Optional[List[_T]] = None) -> _T:
        return self.comm.scatter(objs, src)  # type: ignore

    def abort(self):
        self.comm.Abort(-1)

    def barrier(self):
        self.comm.Barrier()


_dist_envs: Dict[str, DistEnv] = {}
_runtime_backend: Literal['cpu', 'cuda', None] = None


def set_runtime_dist_env_backend(
    backend: Literal['cpu', 'cuda']
):
    logger.info(f"Set runtime dist env backend type to {backend}")

    global _runtime_backend
    _runtime_backend = backend


def _get_or_init_dist_env(backend: Literal['cpu', 'cuda']) -> DistEnv:
    """
    If EASIER JIT backend is CUDA:
    -   We rely on `torch.distributed` with CUDA backend that encapsulates NCCL
        for runtime halo exchange.
        This requires environment variables like WORLD_SIZE RANK MASTER_ADDR
        MASTER_PORT etc. to be set correctly.

    If EASIER JIT backend is CPU:
    -   We rely on a MPI-based DistEnv for runtime halo exchange.

    P.S. About MPIRUN:
    For diagnostic purposes, we may launch EASIER programs with `mpirun`
    where no environment variables like RANK or MASTER_ADDR exist.
    In such cases, we can omit the wiring-up process for MPI communicator and
    directly use MPI_COMM_WORLD, and EASIER CPU backend is immediately ready.
    """
    dist_env = _dist_envs.get(backend, None)
    if dist_env is None:
        if os.getenv('EASIER_USE_MPIRUN') is not None:
            # Allow directly launching with `mpirun`, for debug purposes.
            comm: MPI.Intracomm = MPI.COMM_WORLD
            os.environ["WORLD_SIZE"] = str(comm.size)
            os.environ["RANK"] = str(comm.rank)

            # NOTE MPI standard doesn't define the concept "local rank" and
            # mpi4py doesn't offer such an API either.
            # Read OpenMPI-specific environment variable instead.
            os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]

            if not ("MASTER_ADDR" in os.environ and
                    "MASTER_PORT" in os.environ):
                # The MASTER_ADDR:MASTER_PORT pair is only needed by
                # NCCL communication backend via `torch.distributed`.
                logger.warning(
                    "Environment variables MASTER_ADDR and MASTER_PORT are"
                    " not set under `mpirun`"
                    " (e.g."
                    " `mpirun -x MASTER_ADDR=ip -x MASTER_PORT=port ...`)"
                    "\n"
                    "'localhost:23456' will be set for single-node"
                    " multiprocessing."
                    " For multi-node, please set them correcly."
                )
                os.environ["MASTER_ADDR"] = "localhost"
                os.environ["MASTER_PORT"] = "23456"

            if backend == 'cpu':
                dist_env = MPIDistEnv()
            elif backend == 'cuda':
                dist_env = TorchDistEnv('cuda')
            else:
                raise EasierJitException(f"Unsupported JIT backend {backend}")

        elif is_launched_by_easier_launcher():
            # easier launcher would inject env vars like torchrun does,
            # e.g. RANK, LOCAL_RANK, LOCAL_WORLD_SIZE
            # see `/easier/launcher/setup_template.sh`
            if backend == 'cpu':
                dist_env = MPIDistEnv()
            elif backend == 'cuda':
                dist_env = TorchDistEnv('cuda')
            else:
                raise EasierJitException(f"Unsupported JIT backend {backend}")

        else:
            if backend == 'cpu':
                dist_env = MPIDistEnv(local_rank=0)
            elif backend == 'cuda':
                dist_env = DummyDistEnv(backend)
            else:
                raise EasierJitException(f"Unsupported JIT backend {backend}")

        _dist_envs[backend] = dist_env

    return dist_env


def get_runtime_dist_env() -> DistEnv:
    """
    Get the DistEnv instance for communication during JIT runtime.
    This instance can be retrieved only during or after JIT process.
    """
    if _runtime_backend is None:
        raise EasierJitException("The backend for runtime isn't set")
    return _get_or_init_dist_env(_runtime_backend)


def get_cpu_dist_env() -> DistEnv:
    """
    Get the DistEnv instance for communication between CPUs.
    This instance is ready from the very beginning.
    """
    return _get_or_init_dist_env('cpu')


def get_mpi_communicator():
    return MPI.COMM_WORLD
