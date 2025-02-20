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
        raise AssertionError(
            e,
            "To see exception details, run unit tests in the command line with"
            "`pytest -s tests/.../test.py::test_func`")


def mpirun_singlenode(nprocs: int, func, args=(), kwargs={}):
    """
    mpi4py executor won't record call stack for us, so it's recommended to
    add concrete failure message on each assertion for locating failures.

    To see exception details, run unit tests in the command line with
    `pytest -s tests/.../test.py::test_func` where `-s` captures stderr.
    """
    from mpi4py.futures import MPIPoolExecutor
    with MPIPoolExecutor(nprocs, env={
        "EASIER_USE_MPIRUN": "1",
        "EASIER_LOG_LEVEL": "DEBUG"
    }) as pool:
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
