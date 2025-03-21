# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import random
import string
from typing import Callable, List, Literal, Tuple
from unittest.mock import patch
import os
import pytest

import torch
import torch.distributed
from torch.multiprocessing.spawn import spawn

import easier.core.runtime.dist_env as _DM
from easier.core.utils import get_random_str
from easier import init

have_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="no CUDA"
)

when_ngpus_ge_2 = pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="no enough CUDA GPU (ngpus >= 2) to test distribution"
)


def detect_cuda_aware_openmpi_e2e_setting() -> Tuple[bool, str]:
    """
    Returns:
    -   bool: if there is CUDA-aware OpenMPI
    -   str: what component of the CUDA-aware OpenMPI for testing is missing
    """
    try:
        import mpi4py
    except ImportError:
        return False, "mpi4py is not installed"

    if torch.cuda.device_count() < 2:
        return False, "CUDA device number < 2"

    if not torch.distributed.is_mpi_available():
        return False, "torch is not built with MPI distributed backend"

    import sys
    if sys.platform != 'linux':
        return False, "OS != linux"

    # expect to see
    # ```shell
    # $ ompi_info --parsable --all | grep mpi_built_with_cuda_support:value
    # mca:mpi:base:param:mpi_built_with_cuda_support:value:true
    # ```
    import subprocess
    ompi_info_ret: bytes = subprocess.check_output(
        "ompi_info --parsable --all | grep mpi_built_with_cuda_support:value",
        shell=True
    )
    if b'cuda_support:value:true' not in ompi_info_ret:
        return False, "no CUDA-aware OpenMPI"

    return True, "SHOULD NOT PRINT THIS"


has_mpi_e2e, mpi_e2e_missing_dep = \
    detect_cuda_aware_openmpi_e2e_setting()

mpi_e2e: List[pytest.MarkDecorator] = [
    pytest.mark.skipif(
        not has_mpi_e2e,
        reason="MPI end-to-end tests need to be run with PyTorch built with"
        " CUDA-aware MPI and in a distributed CUDA environment,"
        f" but we get: {mpi_e2e_missing_dep}"
    ),
    pytest.mark.mpi_e2e  # test group name, can be run by `pytest -m mpi_e2e`
]


def _torchrun_spawn_target(
    local_rank: int, world_size: int, func, args, kwargs,
    init_type: Literal['none', 'cpu', 'cuda']
):
    os.environ["EASIER_LOG_LEVEL"] = "DEBUG"

    # Fake torchrun env vars
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)

    # Fake torch.distributed.is_torchelastic_launched() to return True
    os.environ["TORCHELASTIC_RUN_ID"] = "EASIER_UNIT_TEST_RUN"

    if init_type in ['cpu', 'cuda']:
        init(
            'gloo' if init_type == 'cpu' else 'nccl',
            init_method='tcp://localhost:24689',
            world_size=world_size,
            rank=local_rank
        )
        _DM.set_dist_env_runtime_device_type(init_type)

    else:
        assert init_type == 'none'

    func(local_rank, world_size, *args, **kwargs)


def torchrun_singlenode(
    nprocs: int, func, args=(), kwargs={},
    init_type: Literal['none', 'cpu', 'cuda'] = 'cpu'
):
    """
    To see exception details, run unit tests in the command line with
    `pytest -s tests/.../test.py::test_func` where `-s` captures stderr.

    Args:
    - init_type:
        'none': no special initialization is done, a worker process is just
            like a brand new torchrun subprocess, with only env vars set.
        'cpu': call easier.init('gloo')
        'cuda': call easier.init('nccl')
            test cases specifying 'cuda' should take care of the minimum CUDA
            device number. e.g. using
            `pytest.mark.skipif(torch.cuda.device_count() < 2, reason='')`
    """
    spawn(
        _torchrun_spawn_target,
        (nprocs, func, args, kwargs, init_type),
        nprocs=nprocs,
        join=True
    )


def _mpirun_spawn_target(
    func, func_args_kwargs,
    init_type: Literal['none', 'cpu', 'cuda']
):
    (args, kwargs) = func_args_kwargs

    os.environ["EASIER_LOG_LEVEL"] = "DEBUG"

    from mpi4py import MPI
    local_rank = MPI.COMM_WORLD.rank
    world_size = MPI.COMM_WORLD.size

    if init_type in ['cpu', 'cuda']:
        init('mpi')
        _DM.set_dist_env_runtime_device_type(init_type)

    else:
        assert init_type == 'none'

    try:
        func(local_rank, world_size, *args, **kwargs)
    except Exception as e:
        import traceback
        from easier import logger
        logger.error(traceback.format_exc())
        raise AssertionError(
            e,
            "To see exception details, run unit tests in the command line with"
            "`pytest -s tests/.../test.py::test_func`"
        )


def mpirun_singlenode(
    nprocs: int, func, args=(), kwargs={},
    init_type: Literal['none', 'cpu', 'cuda'] = 'cpu'
):
    """
    mpi4py executor won't record call stack for us, so it's recommended to
    add concrete failure message on each assertion for locating failures.

    To see exception details, run unit tests in the command line with
    `pytest -s tests/.../test.py::test_func` where `-s` captures stderr.
    """
    try:
        from mpi4py.futures import MPIPoolExecutor
    except ImportError:
        assert False, \
            "Cannot import the optional testing dependency mpi4py," \
            " the test case using this launcher function should be wrapped" \
            " by some pytest.mark.skipif()"

    # main=False: don't pickle and inherit all globals, because the parent
    #   process is launched by pytest.
    with MPIPoolExecutor(nprocs, main=False) as pool:
        futures = []
        for rank in range(nprocs):
            future = pool.submit(
                _mpirun_spawn_target,
                func,
                (args, kwargs),
                init_type=init_type
            )
            futures.append(future)

        for future in futures:
            future.result()  # re-raise AssertException to host test environment


def assert_tensor_list_equal(la: List[torch.Tensor],
                             lb: List[torch.Tensor]):
    assert len(la) == len(lb)
    for a, b in zip(la, lb):
        assert torch.equal(a, b)
