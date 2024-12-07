# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import random
import string
from typing import List
from unittest.mock import patch
import os
import pytest

import torch
from torch.multiprocessing import spawn  # type: ignore

import easier.core.runtime.dist_env as _DM
from easier.core.utils import get_random_str

have_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")

when_ngpus_ge_2 = pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="no enough CUDA GPU (ngpus >= 2) to test distribution")


def _torchrun_spawn_target(local_rank: int,  # 1st arg is added by torch.spawn
                           world_size: int, func, args, kwargs):

    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)

    orig_dist_env_ctor = _DM.TorchDistEnv.__init__

    def _tcp__init__(self: _DM.TorchDistEnv, backend):
        orig_dist_env_ctor(self, backend,
                           torch_dist_init_kwargs={
                               'init_method': 'tcp://localhost:24689',
                               'world_size': world_size,
                               'rank': local_rank
                           })

    # We are not really launching `torchrun` therefore we need to enforce
    # the initialization of `TorchDistEnv` with tcp:localhost init_method
    with patch('torch.distributed.is_torchelastic_launched',
               new=lambda: True), \
        patch(f'{_DM.__name__}.{_DM.TorchDistEnv.__name__}.__init__',
              new=_tcp__init__):

        func(local_rank, world_size, *args, **kwargs)


def torchrun_singlenode(nprocs: int, func, args=(), kwargs={}):
    spawn(_torchrun_spawn_target, (nprocs, func, args, kwargs), nprocs=nprocs)


def _mpirun_spawn_target(func, args, kwargs):
    from mpi4py import MPI
    local_rank = MPI.COMM_WORLD.rank
    world_size = MPI.COMM_WORLD.size
    try:
        func(local_rank, world_size, *args, **kwargs)
    except Exception as e:
        import traceback
        from easier import logger
        logger.error(traceback.format_exc())
        raise


def mpirun_singlenode(nprocs: int, func, args=(), kwargs={}):
    """
    mpi4py executor won't record call stack for us, so it's recommended to
    add concrete failure message on each assertion for locating failures.
    """
    from mpi4py.futures import MPIPoolExecutor
    with MPIPoolExecutor(nprocs, env={"EASIER_USE_MPIRUN": "1"}) as pool:
        futures = []
        for rank in range(nprocs):
            future = pool.submit(_mpirun_spawn_target, func, args, kwargs)
            futures.append(future)

        for future in futures:
            future.result()  # re-raise AssertException to host test environment


def assert_tensor_list_equal(la: List[torch.Tensor],
                             lb: List[torch.Tensor]):
    assert len(la) == len(lb)
    for a, b in zip(la, lb):
        assert torch.equal(a, b)
